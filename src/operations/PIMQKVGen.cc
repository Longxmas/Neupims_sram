// (1, E) * (3, E)

#include "PIMQKVGen.h"

PIMQKVGen::PIMQKVGen(std::string name) : Operation(name) {}

std::vector<Ptr<BTensor>> PIMQKVGen::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    // inputs: (1, E)
    _batch_size = inputs.size();
    assert(_batch_size == 1);
    spdlog::info("PIMQKVGen (batch size): {}", _batch_size);

    _nh = _config.model_n_head;
    _dk = _config.model_n_embd / _config.model_n_head;
    _E = _config.model_n_embd;
    spdlog::info("PIMQKVGennh:{}, dk:{}", _nh, _dk);
    
    assert((inputs.size() == 2 && _inputs.size() == 2) ||
           (inputs.size() == 1 && _inputs.size() == 3));

    for (size_t i = 0; i < inputs.size(); ++i) {
        _inputs[i] = inputs[i];
        spdlog::info("PIMQKVGen input idx: {} / input sz: {}", i, inputs[i]->get_dims());
    }

    _outputs.resize(_batch_size);

    // assert(inputs.size() == 2);
    auto input0_dims = _inputs[0]->get_dims();
    auto input1_dims = _inputs[1]->get_dims();
    assert(*input0_dims.rbegin() == *(input1_dims.rbegin() + 1));

    auto larger_dim = input0_dims.size() > input1_dims.size() ? input0_dims : input1_dims;
    std::vector<uint32_t> output_dims(larger_dim.begin(), larger_dim.end());
    *(output_dims.rbegin() + 1) = *(input0_dims.rbegin() + 1);  // Set (1, x) in matmul (1, K) x (K, N).
    *output_dims.rbegin() = *input1_dims.rbegin();  // Set (x, N) in matmul (1, K) x (K, N)
    spdlog::info("PIMQKVGen output sz: {}", output_dims);

    _outputs[0] =
        std::make_shared<NPUTensor>(_name + "_output", output_dims, NPUTensorBufType::ACT, false);

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    spdlog::info("input0 : {}  / input1: {} / output0 : {}", input0_dims, input1_dims, output_dims);

    return _outputs;
}

void PIMQKVGen::initialize_tiles() { _tiles.push_back(initialize_instructions()); }

Tile PIMQKVGen::initialize_instructions() {
    uint32_t banks_per_channel = 32;  // FIXME:

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };

    addr_type sram_act_base = SPAD_BASE;
    addr_type sram_acc_base = ACCUM_SPAD_BASE;
    addr_type sram_addr = sram_act_base;
    addr_type sram_acc_addr = sram_acc_base;
    
    auto activation_tensor = std::static_pointer_cast<NPUTensor>(_inputs[0]);
    auto weight_tensor = std::static_pointer_cast<NPUTensor>(_inputs[1]);

    int counter_for_debug = 0;  // delete it

    uint32_t ch = _inputs[1].get_channel();
    uint32_t height = 3 * _E;
    std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;
    uint32_t tiles_per_chunk =
        height / banks_per_channel;  // number of comp-readres kernels

    for (int chunk = 0; chunk < _chunks; chunk++) {
        // uint64_t make_address(channel, rank, bankgroup, bank, row, col);
        // uint64_t encode_pim_header(channel, row, bool for_gwrite, num_comps, num_readres);

        uint64_t input_row = 0;  // FIXME: decode row index from dram address
        std::pair<addr_type, uint32_t> sram_entry_for_gw = allocate_sram_addr(0, false);
        uint64_t gwrite_addr =
            AddressConfig::make_address(ch, 0, 0, 0, input_row, 0);  // FIXME: real gwrite addr
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_GWRITE,
            .dest_addr = sram_entry_for_gw.first,
            .size = 0,
            .src_addrs = std::vector<addr_type>{gwrite_addr},  // FIXME: gwrite addr
            .operand_id = _INPUT_OPERAND,
        });
        // GWRITE (channel, bank, row)

        for (int ti = 0; ti < tiles_per_chunk; ti++) {
            std::pair<addr_type, uint32_t> sram_entry = allocate_sram_addr(0, false);
            addr_type sram_addr_phdr = sram_entry.first;
            int num_head_in_tile =
                (chunk == _chunks - 1) ? _heads_in_last_chunk : _heads_per_tile;
            // 假设_chunks是列数，tiles_per_chunk是行数
            // DRAM_row 就是第ti行，第chunk列
            uint32_t DRAM_row = key->_rows[ti * _chunks + chunk];
            int num_comps = _comps_per_head * num_head_in_tile; // 该tile需要执行的计算指令数量
            int num_readres = num_head_in_tile; // readres是read result的意思，每个头的结果需要一个read指令

            if (num_head_in_tile == 0) {
                spdlog::info("num_head_in_tile must be greater than 0!!!");
                exit(-1);
            }
            uint32_t p_header_addr =
                AddressConfig::encode_pim_header(ch, DRAM_row, false, num_comps, num_readres);
            // P_HEADER (num_comps = comps_per_head * num_heads, num_readres
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_HEADER,
                .dest_addr = sram_addr_phdr,
                .size = 0,
                .src_addrs = std::vector<addr_type>{p_header_addr},
                .operand_id = _INPUT_OPERAND,
            });

            std::string cmds = "P_HEADER ";

            for (int head = 0; head < num_head_in_tile; head++) {
                // 这个地址怎么算的？不需要考虑tile维度？对应第chunk列的第head个头
                int hi = _heads_per_tile * chunk + head;
                uint64_t dram_addr = AddressConfig::encode_pim_comps_readres(
                    ch, DRAM_row, _comps_per_head, head == num_head_in_tile - 1);
                auto sram_entry = allocate_sram_addr(banks_per_channel, false);
                addr_type sram_addr = sram_entry.first;
                if (_config.dram_type == DramType::NEWTON) {
                    Instruction comp_inst = Instruction{
                        .opcode = Opcode::PIM_COMP,
                        .dest_addr = sram_addr,
                        .size = 0,
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    };
                    // spdlog::info("comps:{}", _comps_per_head);
                    // 每个head执行comps_per_head条计算指令
                    for (int j = 0; j < _comps_per_head; j++) {
                        // COMP * comps_per_head (channnel, row)
                        tile.instructions.push_back(comp_inst);
                        cmds += "COMP ";
                    }
                    // 每个head用一条指令进行结果的读取
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_READRES,
                        .dest_addr = sram_addr,
                        // 读取粒度为bank? 将该channel的所有Bank的值读取到SRAM里？
                        .size = sram_entry.second,  // ??? 
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                    cmds += "READRES ";

                } else {
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::PIM_COMPS_READRES,
                        .dest_addr = sram_addr,
                        .size = sram_entry.second,  // ???
                        .src_addrs = std::vector<addr_type>{dram_addr},
                        .operand_id = _INPUT_OPERAND,
                    });
                }

                if (sram_readres_addrs.find(hi) == sram_readres_addrs.end())  // not exists
                    sram_readres_addrs[hi] = std::vector<addr_type>{sram_addr};
                else
                    sram_readres_addrs[hi].push_back(sram_addr);
            }
        }
    }

    spdlog::info("pim_gemv MOVOUT begin, _nh: {}", _nh);
    for (int hi = 0; hi < _nh; hi++) {
        assert(sram_readres_addrs[hi].size() == tiles_per_chunk);
        uint32_t column_height = 3 * _E;  // tiles_per_chunk * banks_per_channel;
        std::pair<addr_type, uint32_t> sram_acc_entry = allocate_sram_addr(column_height, true);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::DUMMY,  // for buffer.check_hit (src_addrs)
            .dest_addr = sram_acc_entry.first,
            .size = column_height,
            .src_addrs = sram_readres_addrs[hi],
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_acc_entry.first,
            .size = sram_acc_entry.second,
            .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                ->_inners[hi]
                                ->get_all_addrs(),
            .operand_id = _OUTPUT_OPERAND,
        });
        // sram_acc_base += column_height * _config.precision;
        counter_for_debug++;
    }
    
    std::string yellow = "\033[1;33m";
    spdlog::info("{}MOVOUT size: {}\033[0m", yellow, counter_for_debug);

    return tile;
}

void PIMQKVGen::calculate_loops() {
    assert(sram_size_needed() < _config.spad_size KB / 2);

    uint32_t E = _config.model_n_embd;
    uint32_t page_size = _config.dram_page_size / _config.precision;
    uint32_t banks_per_channel = _config.dram_banks_per_ch;
    uint32_t datas_per_comp_cmd = _config.pim_comp_coverage; // newton --> 16

    _E = E;
    // 对于7B模型，_chunks = 4096 / 512 = 8，意味着一个激活需要分配到8个sram里进行计算
    _chunks = ceil((double)E / page_size);            // # of gwrite
    // 对于7B模型，_dk=128, page_size = 1024 / 2 = 512, _heads_per_tile = 4
    _heads_per_tile = ceil((double)page_size / _dk);  // # of readres 
    _heads_in_last_chunk = ceil((double)(E % page_size) / _dk);
    // for debug
    spdlog::info("E: {}, page_size: {}, _dk: {}", E, page_size, _dk);
    spdlog::info("heads_in_last_chunk: {}", _heads_in_last_chunk);

    // 对于7B模型，_comps_per_head = 8, 意味着每个头对应的维度需要8条计算指令执行？
    _comps_per_head = ceil((double)_dk / datas_per_comp_cmd);

    std::string yellow = "\033[1;33m";
    std::string reset = "\033[0m";
    spdlog::info("{}chunks:{}, heads_per_tile:{}, comps_per_head:{} {}", yellow, _chunks,
                 _heads_per_tile, _comps_per_head, reset);
}

uint32_t PIMQKVGen::sram_size_needed() {
    return 3 * _E * _config.model_n_head * _config.precision;
}