# From the Warp to the Token Engine

Most GPU education is wrong in one of two ways. It either traps you in toy CUDA or it hands you libraries first and calls that understanding.

This path keeps only the highest-leverage pieces. Pick one box, one model family, one dtype, and one serving target at the start, then refuse to get generic later.

A good default is `4090 or H100`, `Llama-3.1-8B-class dense decoder`, `BF16`, and `single-GPU decode latency plus respectable throughput`. The end artifact is a token engine whose hot path you actually own because you understand the machine below it.

## Section 1: Product: Freeze the battlefield -- 0.5 weeks

- Freezing product constraints (`decision`, `0`) -- Pick one checkpoint format, one architecture family, one KV layout target, and one serving objective, then freeze them. Industrial leverage comes from going deep on one real shape regime, not from being vaguely compatible with everything.
- Instrumentation (`ncu`, `nsys`, `cuobjdump`, `0`) -- Treat the profiler as part of the machine, not as optional observability. `ncu` tells you why a kernel lost, `nsys` tells you where the product lost, and disassembly tells you whether the compiler quietly changed the deal.
- Building CPU oracles and a kernel lab (`C++ + CUDA C++`, `350`) -- Write tiny reference implementations for transpose, reduction, softmax, RMSNorm, and causal attention, then build one binary that runs kernels by name, checks results, warms up, times medians, and emits CSV. Without a stable harness and unquestioned correctness oracle, later fused kernels turn into storytelling.

## Section 2: Machine: What is a warp really doing? -- 1 week

- Building a tiny SIMT machine (`C`, `300`) -- Build 32 lanes, active masks, predication, divergence, reconvergence, a toy shared memory, and a toy scheduler. If you cannot run a warp in software, the real hardware stays mystical and every later optimization becomes folklore.
- Coding a tiny kernel ISA + assembler (`Python`, `250`) -- You need a level low enough that masks, branches, and memory traffic are explicit, but not so low that you disappear into hardware trivia. This is the cleanest way to force yourself to think in execution masks and explicit movement.
- Building a transaction model (`C`, `200`) -- Feed 32 addresses in and get sectors, transactions, wasted bytes, and bank conflicts out. This is one of the rarest pieces of practical understanding in GPU work, because most people recite “coalescing” without owning the mechanism.

## Section 3: Kernels: Own movement and reductions -- 1.5 weeks

- Building a transpose (`CUDA C++`, `250`) -- This is the first real CUDA kernel because coalescing and shared-memory banking meet here in a clean way. `vector add` teaches almost nothing, while transpose already starts to look like real kernel work.
- Building staged tile movement (`CUDA C++`, `250`) -- Build disciplined global-to-shared movement for the exact tile shapes later kernels will need, including predicated edges and vectorized paths. Do not build a generic tensor abstraction here, because the point is to own movement before you try to hide it.
- Building warp and block reductions (`CUDA C++`, `300`) -- Build `max` and `sum` reductions at warp scope and block scope, first naively and then with warp intrinsics. Softmax, norms, sampling, and attention all collapse onto these same reduction atoms, so this is not a side exercise.

## Section 4: Compiler: What is a tensor program anyway? -- 1.5 weeks

- Building a tiny tensor IR (`Python`, `500`) -- Support shapes, strides, views, pointwise ops, reductions, and matmul nodes, then stop. The point is not to build PyTorch badly, it is to own the graph that will eventually tell you where custom kernels are required.
- Building a scheduler (`Python`, `350`) -- Fuse the obvious pointwise chains and keep the hard boundaries visible. This is where you learn that the real enemy is usually not compute, but bad graph-shaped memory traffic.
- Emitting simple CUDA kernels (`Python`, `300`) -- Generate CUDA for elementwise and reduction kernels directly from the IR and run them through the same harness. Ugly codegen is fine here, because ownership of the lowering path matters more than elegance.

## Section 5: Product: Make the model talk -- 2 weeks

- Building a checkpoint loader (`C++`, `350`) -- Load one checkpoint format, one architecture, and one dtype, and reject everything else. The goal is to get weights into device memory with a layout your kernels want, not to build a universal frontend.
- Building a token engine (`C++`, `500`) -- Build a minimal inference runtime that owns tensor storage, stream usage, execution order, workspace reuse, and failure handling. Until tokens come out, you do not have a product, only a kernel collection.
- Building a KV cache (`C++/CUDA C++`, `300`) -- Build the data structure, update path, and address math for K/V storage across sequence steps. Make the layout explicit and frozen, because later attention performance depends more on this choice than on abstract cleverness.
- Building decode glue (`CUDA C++`, `450`) -- Build RMSNorm, RoPE, KV update, logits postprocessing, and sampling kernels, and wire them into the token engine. Real decode latency is often destroyed by “small” kernels everyone else treats as glue.
- Building prefill vs decode split (`C++`, `250`) -- Make prefill and decode explicit separate execution modes in the runtime. They are different workloads with different shape regimes and product goals, and pretending otherwise is how people end up with mediocre engines.

## Section 6: Mainloop: Own one honest tensor kernel -- 2 weeks

- Building a fixed-shape projection engine (`CUDA C++`, `1000`) -- Build a shared-memory GEMM for the exact projection shapes your chosen model uses, not a generic GEMM library. This keeps the anti-sheep benefit of writing one honest tensor kernel while preserving industrial taste about real shapes.
- Building a sweep tool (`Python`, `180`) -- Sweep tile shapes, block sizes, vector widths, stages, and shared-memory footprints, then store timing and profiler summaries. This replaces vibes with evidence and teaches you that the search space for one real product is small enough to own if you refuse to be generic.
- Building fused epilogues (`CUDA C++`, `250`) -- Extend the projection engine with one or two fused writeback variants such as bias-add, residual-add, or type conversion. The boundary after compute is part of the kernel and often part of the bottleneck.
- Building a vendor baseline (`cuBLASLt`, `0`) -- Add a path that runs the same projection shapes through `cuBLASLt`. This is not the implementation you are learning from, it is the enemy and the oracle, and you need to know when your handwritten path is educationally useful but industrially wrong.

## Section 7: Attention: Kill the score matrix -- 2 weeks

- Building naive causal attention (`CUDA C++`, `300`) -- Write the obvious correct version first, even though it materializes the score matrix and is badly shaped for the GPU. You need a real victim before you can appreciate why online softmax and fused movement matter.
- Building online softmax (`CUDA C++`, `350`) -- Build the streaming max/sum update over tiles and test it in isolation until the numerics are trustworthy. This is one of the deepest ideas in the whole path, because it turns attention from “huge intermediate buffer” into “real kernel.”
- Building tiled causal attention (`CUDA C++`, `800`) -- Build a blocked QK and PV path over the actual KV-cache layout used by your token engine, using shared-memory staging and online softmax state. The implementation should live inside the product’s real shape assumptions, because a fake standalone attention kernel is a detour, not a milestone.
- Building integrated decode attention (`CUDA C++`, `350`) -- Replace the reference attention path in the token engine with your own fused decode path and make the product emit tokens through it. This is where the build stops being “CUDA study” and becomes an actual industrial artifact.

## Section 8: Frontier: Rewrite the real wound -- 2 weeks

- Building a barrier model (`C`, `250`) -- Write a tiny phase/barrier simulator with producer-consumer stages, arrivals, waits, and completion. This is the cleanest way to make `mbarrier` semantics obvious without first drowning in NVIDIA terminology.
- Building an async pipeline (`CUDA C++`, `450`) -- Take one real hot kernel from the engine and rewrite it with async copies, barriers, and double buffering. The point is not to “cover `cp.async`,” but to make movement and compute overlap in a product-shaped kernel where the profiler proves overlap matters.
- Inspecting emitted instructions (`PTX/SASS`, `150`) -- Verify that the compiler actually gave you the path you think you wrote. Eventually you need to know what the compiler really asked the GPU to do, not what the CUDA source politely suggested.
- Building a tensor-core or Hopper path (`CUDA C++/PTX`, `600`) -- Add a tensor-core path for the exact shapes that dominate the chosen product, not a general tensor-core framework. If the card is Hopper, this is where `TMA` and `WGMMA` enter; if it is not, stay honest and push the best path your card actually supports.
- Reading `CUTLASS` and `CuTe` (`C++`, `0`) -- Only now do you read them. At this point you can steal abstractions from strong engineers instead of outsourcing your understanding to them.

## What gets cut

- `vector add` -- Fine as a smoke test and useless as a milestone. If a build does not unlock the next build, it is out.
- Copy zoo -- Plain copy variants are useful for a sanity check but too weak to be a centerpiece. `transpose` is the movement kernel worth respecting because coalescing and shared-memory behavior collide there.
- Genericity theater -- Universal frontends, generic GEMM frameworks, and compatibility with every model are seductive ways to avoid learning the real shape regime. Industrially, narrow and deep beats wide and vague.
- `WMMA` tourism -- Calling one tensor-core API does not teach modern compute kernels. Modern kernel work is about feeding the machine, not touching the machine once.
- `CUTLASS` first -- Starting with vendor abstractions gives you vocabulary before judgment. That is exactly how people become library monkeys.

## Compressed verdict

- `C` owns the machine model because it is the cleanest way to make execution, synchronization, and transactions explicit.
- `Python` owns the compiler, sweep tools, and runtime glue because you need to control the graph and its search space, not just hand-write kernels forever.
- `CUDA C++` owns the first honest kernels because you need one direct encounter with the machine before async or Hopper-specific machinery.
- `PTX` and `SASS` own the truth when abstractions leak because eventually you need to verify what the compiler actually asked the GPU to do.
- The capstone is a token engine with owned decode attention, not a benchmark chart of isolated kernels.

If this path works, Eric will own what most engineers quietly take for granted: warp execution, transaction formation, shared-memory behavior, online softmax, KV-cache layout, decode/prefill split, emitted instructions, and the exact point where library composition stops working. That is the kind of understanding that prevails in industrial production.
