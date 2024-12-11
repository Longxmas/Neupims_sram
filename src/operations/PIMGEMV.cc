// for PIM baseline without kernel fusion

#include "PIMGEMV.h"

PIMGEMV::PIMGEMV(std::string name) : Operation(name) {}

std::vector<Ptr<BTensor>> PIMGEMV::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _inputs = inputs;

    _batch_size = inputs.size() / 2;

    spdlog::info("PIMGEMV (batch size): {}", _batch_size);

    uint32_t i = 0;
    for (auto tensor : inputs) {
        if (i < _batch_size) {
            _qs.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        } else {
            _ks.push_back(std::static_pointer_cast<PIMTensor>(tensor));
        }
        i++;
    }

    _outputs.resize(_batch_size);

    _nh = _qs[0]->get_dims()[0];
    _dk = _qs[0]->get_dims()[2];

    // assert(inputs.size() == 2);
    for (int i = 0; i < _batch_size; ++i) {
        auto Q = _qs[i];  // [h, 1, d_k]
        auto K = _ks[i];  // [h, d_k, seq_len]

        uint32_t seq_len = K->get_dims()[2];

        // d_k of Q == d_k of K^T
        spdlog::info("Q: {}, K: {}", Q->get_dims(), K->get_dims());
        assert(Q->get_dims()[2] == K->get_dims()[1]);
        std::vector<uint32_t> gemv_output_dim{_nh, 1, seq_len};

        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output", gemv_output_dim,
                                                  NPUTensorBufType::ACT, false);
    }

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    // spdlog::info("input dims : {} {}", Q->get_dims(), K->get_dims());
    spdlog::info("output dim : {}", _batch_size);

    return _outputs;
}

void PIMGEMV::initialize_tiles() { _tiles.push_back(initialize_instructions()); }

Tile PIMGEMV::initialize_instructions() {
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

    int counter_for_debug = 0;  // delete it

    // 对于batch内的请求，逐个执行计算
    for (int i = 0; i < _batch_size; i++) {
        auto query = _qs[i];
        auto key = _ks[i];

        uint32_t ch = key->get_channel();
        std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;

        uint32_t tiles_per_chunk =
            key->get_allocated_seq_len() / banks_per_channel;  // number of comp-readres kernels

        for (int chunk = 0; chunk < _chunks; chunk++) {
            // uint64_t make_address(channel, rank, bankgroup, bank, row, col);
            // uint64_t encode_pim_header(channel, row, bool for_gwrite, num_comps, num_readres);

            uint64_t query_row = 0;  // FIXME: decode row index from dram address
            uint64_t p_header_addr = AddressConfig::encode_pim_header(ch, query_row, true, 0, 0);
            //  P_HEADER (for_gwrite=true)
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_HEADER,
                .dest_addr = sram_addr,
                .size = 0,
                .src_addrs = std::vector<addr_type>{p_header_addr},
                .operand_id = _INPUT_OPERAND,
            });
            // 写入数据的维度，维度呢？
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_GWRITE,
                .dest_addr = sram_addr,
                .size = 0,
                .src_addrs = std::vector<addr_type>{p_header_addr},  // FIXME: gwrite addr
                .operand_id = _INPUT_OPERAND,
            });
            // GWRITE (channel, bank, row)

            for (int ti = 0; ti < tiles_per_chunk; ti++) {
                int num_head_in_tile =
                    (chunk == _chunks - 1) ? _heads_in_last_chunk : _heads_per_tile;
                // 假设_chunks是列数，tiles_per_chunk是行数
                // DRAM_row 就是第ti行，第chunk列
                uint32_t DRAM_row = key->_rows[ti * _chunks + chunk];
                int num_comps = _comps_per_head * num_head_in_tile; // 该tile需要执行的计算指令数量
                int num_readres = num_head_in_tile; // readres是read result的意思，每个头的结果需要一个read指令
                p_header_addr =
                    AddressConfig::encode_pim_header(ch, DRAM_row, false, num_comps, num_readres);
                // P_HEADER (num_comps = comps_per_head * num_heads, num_readres
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_HEADER,
                    .dest_addr = sram_addr,
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
                            .size = banks_per_channel * _config.precision,  // ??? 
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        });
                        cmds += "READRES ";

                    } else {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::PIM_COMPS_READRES,
                            .dest_addr = sram_addr,
                            .size = banks_per_channel * _config.precision,  // ???
                            .src_addrs = std::vector<addr_type>{dram_addr},
                            .operand_id = _INPUT_OPERAND,
                        });
                    }

                    if (sram_readres_addrs.find(hi) == sram_readres_addrs.end())  // not exists
                        sram_readres_addrs[hi] = std::vector<addr_type>{sram_addr};
                    else
                        sram_readres_addrs[hi].push_back(sram_addr);

                    sram_addr += banks_per_channel * _config.precision; // 每个head用一个channel所有的bank处理
                }
            }
        }

        spdlog::info("pim_gemv MOVOUT begin, _nh: {}", _nh);
        for (int hi = 0; hi < _nh; hi++) {
            assert(sram_readres_addrs[hi].size() == tiles_per_chunk);
            uint32_t column_height =
                key->_seq_len;  // tiles_per_chunk * banks_per_channel * _config.precision;
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::DUMMY,  // for buffer.check_hit (src_addrs)
                .dest_addr = sram_acc_base,
                .size = column_height,
                .src_addrs = sram_readres_addrs[hi],
            });
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_acc_base,
                .size = column_height * _config.precision,
                .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                                 ->_inners[hi]
                                 ->get_all_addrs(),
                .operand_id = _OUTPUT_OPERAND,
            });
            sram_acc_base += column_height * _config.precision;
            counter_for_debug++;
        }
    }

    std::string yellow = "\033[1;33m";
    spdlog::info("{}MOVOUT size: {}\033[0m", yellow, counter_for_debug);

    return tile;
}

void PIMGEMV::calculate_loops() {
    assert(sram_size_needed() < _config.spad_size KB / 2);

    uint32_t E = _config.model_n_embd;
    uint32_t page_size = _config.dram_page_size / _config.precision;
    uint32_t banks_per_channel = _config.dram_banks_per_ch;
    uint32_t datas_per_comp_cmd = _config.pim_comp_coverage; // newton --> 16

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

uint32_t PIMGEMV::sram_size_needed() {
    //  calculate total sequence length in batch
    uint32_t total_seq_len = 0;

    for (auto key : _ks) {
        total_seq_len += key->get_dims()[2];
    }

    uint32_t need_size = total_seq_len * _config.model_n_head * _config.precision;

    return need_size;
}