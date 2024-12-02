1. 为什么_pim_acc_spad和_pim_spad不需要cycle()？
NeuPIMSystolicWS重写了cycle()函数。pim相关的操作在这里面实现

2. cycle数表示什么？
void NeuPIMSystolicWS::update_stats() {
    if (!_compute_pipeline.empty()) {
        auto parent_tile = _compute_pipeline.front().parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        parent_tile->stat.compute_cycles++;
    }
为什么不管什么计算op,每次只加1？
这个只是统计用，而且并不是指一次cycle就能执行完一个op。
有专门的计算Iteration次数的函数
cycle_type vec_op_iter = calculate_vector_op_iterations(inst.size);
cycle_type add_tree_iter = calculate_add_tree_iterations(inst.size);

3. interconnnet.cc
负责内存访问和调度流程，in_buffer和out_buffer分别存储待传输和传输完成的请求
与core的交互直接通过in_buffer和out_buffer；
与dram的交互通过_mem_req_queue1和_mem_req_queue2，分别处理SA和PIM两种平台的访存请求


内存请求：
当一个内存访问请求（MemoryAccess）产生时，它会被推送到源节点的输入缓冲区（_in_buffers[src]）。这些请求包括目标节点（dest）和请求的内存访问类型（读取、写入等）。
请求的传输：
在每个时钟周期，cycle() 方法会遍历所有节点，检查输入缓冲区是否有请求可以转移。如果请求的 finish_cycle 小于等于当前周期，就可以将请求从源节点转移到目标节点的输出缓冲区或者内存请求队列。
内存请求队列：
内存请求可能会根据处理平台（SA或PIM）被放入不同的内存请求队列（_mem_req_queue1 或 _mem_req_queue2）。这些队列用于将请求传送到相应的内存处理单元。
统计信息：
每当内存请求完成时，update_stat() 方法会更新对应的内存统计信息（如读取、写入、PIM读取等）。


4. scheduler.cc
负责请求的调度，不负责请求的创建（比如模型算子初始化这些），调度包括
* kv cache的分配
* 子批次交错的处理
我们的项目可以只考虑newton版本的dram，也就是单批次处理
这种情况下，每次要么只有_model_program1不是空指针，要么_model_program2不是空指针

void Scheduler::make_program() {
    std::shared_ptr<BatchedRequest> sub_batch_on_sa;  // = std::make_shared<BatchedRequest>(_breq1);
    std::shared_ptr<BatchedRequest> sub_batch_on_pim;  //= std::make_shared<BatchedRequest>(_breq2);
    if (static_cast<int>(_stage) % 2 == 0) {
        // 如果是A(qkv gen1@npu)， C(Pj/FFNs/QKVgen#1 @npu),  E(Pj/FFNs#1 @npu)
        // 那么breaq1就是npu负责的请求批次
        // 在newton情况下只考虑sa部分的计算
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq1);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq2);
    } else {
        // 如果是B(mha @pim)
        // 那么breaq1就是pim负责的请求批次
        // 在newton情况下只考虑pim部分的计算
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq1);
    }
    _model_program1 =
        std::make_unique<StageProgram>(_model, sub_batch_on_sa, StagePlatform::SA, _stage);
    _model_program2 =
        std::make_unique<StageProgram>(_model, sub_batch_on_pim, StagePlatform::PIM, _stage);
    refresh_status1();
    refresh_status2();
}


void Scheduler::refresh_stage() {
    ...
    if (_config.baseline_exp) {
        // >> newton 对应的stage只有A(qkv gen@npu)，B(mha @pim), E(Pj/FFn @npu),  Finish
        if (_stage == Stage::C) _stage = Stage::E;
        if (_stage == Stage::F) _stage = Stage::Finish;
    }
    ...
}



pim_head可能不用
ffn, projection

gelu look up table
adder + mult 作为address查表（需要分配空间 from kv cache space）