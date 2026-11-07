// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "api/allocator_api.h"
#include "common.h"
#include "hash_map.h"
#include "logging.h"
#include "rdma.h"
#include "shared_ptr.h"
#include "simple_vector.h"
#include "vector.h"

#include <mutex>

namespace moodist {

struct SetupComms;
struct IpcMapper;
struct CpuThread;
struct IbCommon;
struct AllGather;
struct ReduceScatter;

struct DynamicAddresses;

enum class Dtype { float32, float64, int32, int64, bfloat16, count };
enum class Reduction { sum, min, max, avg, count };

static constexpr size_t maxConcurrency = 16;
static constexpr size_t maxDevices = 4;
static constexpr size_t maxChunks = 8;
static constexpr size_t maxBlocks = 1024;
struct alignas(64) CpuAddresses {
  uint32_t stepValue;
  int pid;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  size_t bytes;
};

struct DynamicAddressesNoKeys {
  uint32_t opType;
  uintptr_t gatherAddress;
  size_t gatherBytes;
};

struct DynamicAddressesBase : DynamicAddressesNoKeys {
  std::array<uint32_t, maxDevices> gatherKey;
  uint32_t stepValue;
};

struct alignas(128) DynamicAddresses : DynamicAddressesBase {};
static_assert(sizeof(DynamicAddresses) <= 128);

struct alignas(64) CpuDynamicAddresses {
  uint32_t stepValue;
  uint32_t opType;
  uintptr_t gatherAddress;
  size_t gatherBytes;
};

// A rank's compile_op arrival value (see Group::cpuArrivalValues): one RDMA write puts both words, and the reader
// loads both at once (assuming, as the step values do for 4 bytes, that an aligned write of 8 bytes is not torn)
struct alignas(8) ArrivalValue {
  uint32_t stepValue;
  uint32_t opid;
};

struct StreamData {
  size_t cpuThreadIndex = -1;
  StreamData();
  ~StreamData();
};

struct Group {
  size_t rank;
  size_t size;

  int deviceIndex = 0;
  int pid = 0;

  CUcontext cuContext = nullptr;
  CUdevice cuDevice;

  static constexpr size_t maxConcurrency = moodist::maxConcurrency;
  static constexpr size_t maxDevices = moodist::maxDevices;
  static constexpr size_t maxChunks = moodist::maxChunks;
  static constexpr size_t maxBlocks = moodist::maxBlocks;

  std::unique_ptr<SetupComms> setupComms;
  std::unique_ptr<IpcMapper> ipcMapper;

  // The process group's timeout (see setTimeout): how long a wait for other ranks may take, in the setup (which then
  // fails, see SetupComms), for mappings of other ranks' memory (see IpcMapper::checkSpin) and where the main thread
  // waits for the CPU thread (which then ends the process, see waitForCpuThread)
  std::chrono::steady_clock::duration timeout = std::chrono::minutes(30);
  void setTimeout(std::chrono::steady_clock::duration timeout);
  // Waits until the CPU thread sets `done` (the end of a task that the main thread waits for); after the timeout, ends
  // the process (the task may still refer to the caller's memory). what: the task, for the message.
  void waitForCpuThread(std::atomic_uint32_t& done, const char* what);

  std::vector<std::unique_ptr<Rdma>> rdmaDevs;
  size_t numTrueIbDevs = 0;
  bool rdmaSupportsCuda = false;
  std::unique_ptr<AllGather> allGather;
  std::unique_ptr<ReduceScatter> reduceScatter;
  CpuThread* cpuThread = nullptr;

  bool supportsFabric = false;
  bool supportsMulticast = false;

  AllocatedArray cpuOutBuffer;
  AllocatedArray cpuInBuffer;
  // A compile_op kernel's report that its launch cannot go on (see generateKernel), in host memory (kernelError): the
  // NVLS kernel found a tensor that NVLS addresses at another place of the communication pool on another rank, a peer
  // launched the other kernel of the op (the ranks chose differently, see launchCustomOp in processgroup.cc), or a peer
  // failed so. The first thread to find one claims the record (claims), writes it, marks this rank's launch as failed
  // for the peers (in their variant words, see KernelHandle::variantWords), sets ready, and traps (as does every other
  // thread that finds one, once ready is set): the context fails, and the CPU thread prints the record (see
  // kernelErrorMessage).
  struct KernelError {
    uint32_t claims;
    uint32_t ready;
    uint32_t code; // kernelErrorIdentity, kernelErrorVariant or kernelErrorPeerFailed
    uint32_t rank; // the rank that found it
    uint32_t peer; // the other rank
    uint32_t stepValue;
    uint32_t tensor; // kernelErrorIdentity: the index of the tensor (of this rank's op), in its NVLS slot
    uint32_t pad;
    uint64_t local;  // kernelErrorIdentity: this rank's NVLS identity of the tensor; kernelErrorVariant: its launch's
    uint64_t remote; // the peer's (kernelErrorVariant: the peer's variant word, see KernelHandle::variantWords)
  };
  static constexpr uint32_t kernelErrorIdentity = 1;
  static constexpr uint32_t kernelErrorVariant = 2;
  static constexpr uint32_t kernelErrorPeerFailed = 3;
  // The variant word with which a failed launch marks itself for its peers (no launch's variant word: step values
  // advance by 0x1000 per op and stay below 0x80000000)
  static constexpr uint32_t kernelFailedWord = 0xffffffff;
  AllocatedBuffer kernelError;
  bool kernelErrorReported = false; // (by the CPU thread)
  // The message of the kernel error, if any kernel has reported one (see KernelError), else empty
  std::string kernelErrorMessage() const;
  // The signals of compile_op's RDMA transfers, which its kernels wait for (see CustomOpDescriptor::IbRead and
  // IbWrite): 16 bytes per signal index, a 32-bit word per NIC (maxDevices), all of which the CPU threads set to the
  // step value (+ a generation, see compile_op::Graph::Signal) once the data is there (see writeRdmaSignal in
  // cputhread.cc), or once a slot of a ring of the scratch tensor is free (a credit). Aligned to 64 KiB, for a memory
  // registration of its own (dmabuf ranges are in whole host pages).
  AllocatedArray rdmaSignalBuffer;
  static constexpr size_t rdmaSignalBytes = 16;

  AllocatedArray cudaStepValue;
  std::array<PeerArrayRef, 8> peerCudaStepValue;

  AllocatedArray cudaPeerAddresses;
  std::array<PeerArrayRef, 8> peerCudaPeerAddresses;

  AllocatedArray cudaCommsDeviceDataSent;
  AllocatedArray cpuCommsDeviceDataSent;

  AllocatedArray cudaCommsDeviceDataSent2;
  AllocatedArray cpuCommsDeviceDataSent2;

  AllocatedArray cudaProxyReady;
  std::array<PeerArrayRef, 8> peerCudaProxyReady;

  AllocatedArray cudaCopyDone;
  std::array<PeerArrayRef, 8> peerCudaCopyDone;

  AllocatedArray cudaBlockDone;
  std::array<PeerArrayRef, 8> peerCudaBlockDone;

  AllocatedArray cudaMemSync;
  std::array<PeerArrayRef, 8> peerCudaMemSync;

  // // Per-descriptor signal buffer for fine-grained copy synchronization.
  // // Lazily allocated during compile_op. Grows if a new op needs more slots.
  // // Layout: buffer[concurrencyIndex][blockIdx][descriptorIndex] → uint32_t stepValue
  // AllocatedArray cudaDescriptorSignal;
  // std::array<PeerArrayRef, 8> peerCudaDescriptorSignal;
  // Vector<AllocatedArray> oldDescriptorSignalBuffers; // kept alive + reset after growth

  std::vector<AllocatedArray*> buffersToReset;

  IVector<size_t> ipcRanks;
  IVector<size_t> peerIndices;
  IVector<uint8_t> ipcAccess;
  std::vector<std::vector<uint8_t>> peerIpcAccess;
  std::array<size_t, 8> peerMyRemoteIndex;

  Vector<Vector<size_t>> nodeRanks;
  Vector<size_t> rankToNodeIndex;
  Vector<size_t> rankLocalRank;

  Vector<size_t> fabricDomainRanks;

  // NVLS (NVLink SHARP multicast) for compile_op (see generateKernel), set up by setupNvls when the first op that
  // can use it is compiled: two multicast objects per NVLink domain that can use them (of the group's ranks in it,
  // fabricDomainRanks; a multicast object cannot span domains), of which a rank has its domain's. One over the group's
  // communication pool (see commPool), every slab bound at its logical offset (so logical offset o of every rank of
  // the domain is at commMcBase + o): NVLS only stores to and loads from buffers of the communication pool. One over an
  // area for the ops' step counters. The objects, and what is bound to them, outlive the group.
  struct Nvls {
    static constexpr size_t countersBytes = (size_t)2 << 20;
    // The fewest ranks of a domain that any use of NVLS needs (see nvlsReduceMinRanks in ptx_codegen.cc, which checks
    // the thresholds against it): smaller domains get no multicast objects
    static constexpr size_t minDomainRanks = 8;
    int state = 0; // 0: not set up yet, 1: available, -1: unavailable (to this rank, i.e. to its domain)
    uintptr_t commMcBase = 0;
    uintptr_t countersLocal = 0; // the counter area, through the local mapping
    uintptr_t countersMc = 0;    // the counter area, through the multicast mapping
    size_t nextCounters = 0;     // the offset in the counter area of the next op's counters (the same on all ranks)
    // How many launches of this rank used NVLS (a 32-bit counter in the first 64 bytes of the counter area; see the
    // process group option "nvls_launches"), and how many of those ran nvlsReduceLoop (the next 32 bits;
    // "nvls_loop_launches")
    uintptr_t launches = 0;
    uintptr_t loopLaunches = 0;
  };
  Nvls nvls;
  // Collective over all ranks of the group (setupComms). Returns whether NVLS is available to this rank: whether its
  // domain's objects are set up. A domain has none if it has fewer than Nvls::minDomainRanks ranks (a rank alone in
  // its domain, or without a fabric, is in none), if one of its ranks does not support multicast or has no allocator
  // pool, or if the setup fails on one of its ranks; the other domains are not affected.
  bool setupNvls();

  // The group's communication pool (see allocatorCommPoolCreate): buffers that every rank allocates (and frees) in the
  // same order get the same logical offset on every rank, which compile_op ops can then store to with NVLS. Created on
  // first use; never freed (its memory can outlive the group).
  CommPool* getCommPool();
  std::once_flag commPoolOnce;
  CommPool* commPool = nullptr;

  int allocationNode = -1;

  void* mySharedMem = nullptr;
  size_t mySharedMemSize = 0;
  std::array<void*, 8> peerSharedMem;
  DynamicAddresses* localDyns = nullptr;
  CpuDynamicAddresses* cpuLocalDyns = nullptr;
  CpuAddresses* cpuAddresses = nullptr;
  uint32_t* cpuProxyReady = nullptr;

  uint32_t* cpuStepValues = nullptr;
  uint32_t* cpuStepValuesDeviceChunks = nullptr;

  std::atomic_uint32_t* atomicStepValue = nullptr;

  // compile_op's arrival values (see WorkCustom in cputhread.cc): per concurrency slot and rank, the step value and op
  // id of the latest op of the slot that the rank has reached, which it writes into the memory of the ranks that are
  // not its peers in the op. Only compile_op reads them, and only with >=, so a rank can write them without knowing
  // whether the other rank has finished the slot's previous op, which may be a process group collective that waits
  // for an exact value in cpuStepValues.
  ArrivalValue* cpuArrivalValues = nullptr;
  // compile_op's tensor lists (see WorkCustom in cputhread.cc): per concurrency slot and rank, the header of the list
  // that the rank publishes to this one. Process group collectives use localDyns, in the slots of the ranks that the
  // lists come from, but the all-gather ring keeps its neighbor's lists in slots 0 and 1.
  DynamicAddresses* customDyns = nullptr;
  // compile_op's notifications (see CustomOpDescriptor::Notification): per concurrency slot, rank and notification
  // slot, the step value (+ the generation of the use of the slot) of the latest notification of the rank to this one
  // in that slot. In cpuStepValuesDeviceChunks, the process group's all-gather and reduce-scatter rings write step
  // values under the indices of other ranks into their send neighbor's memory without waiting for it to reach the op
  // (the broadcast rings only into ranks that have reached it), so such a ring 16 ops later (on another stream) could
  // tell a rank still in a compile_op that a partial result was ready before it was. (The all-gather ring writes a
  // rank's index only once that rank is in the ring op: that rank has then finished the compile_op, which it does only
  // once the reads from it are over; the reduce-scatter ring, unused, writes its first one at once.)
  static constexpr size_t numNotificationSlots = 64;
  uint32_t* customNotifications = nullptr;

  AllocatedArray cudaStepValuesBuffer;
  AllocatedArray cudaStepValuesDeviceChunksBuffer;

  HashMap<CUstream, std::unique_ptr<StreamData>> streamData;
  std::array<std::unique_ptr<StreamData>, maxConcurrency> cpuStreamData;

  std::string name;

  std::atomic_size_t refcount = 1;

  template<typename T>
  size_t getSharedOffset(T* myVar) const {
    size_t offset = (uintptr_t)(void*)myVar - (uintptr_t)mySharedMem;
    CHECK(offset + sizeof(T) <= mySharedMemSize);
    return offset;
  }

  template<typename T>
  T* getPeerVar(size_t peerIndex, T* myVar) const {
    return (T*)((uintptr_t)peerSharedMem[peerIndex] + getSharedOffset(myVar));
  }

  Group(size_t rank, size_t size);
  ~Group();

  void init(Function<void()> f, Function<void()> pghandle);
  Function<void()> init2;

  size_t getPeerIndex(size_t rank) const {
    for (size_t i = 0; i != ipcRanks.size(); ++i) {
      if (ipcRanks[i] == rank) {
        return i;
      }
    }
    throw std::runtime_error("Moodist could not find a peer rank which should be on the same node but isn't. This "
                             "could indicate that the local machine topology is not supported by Moodist.");
  }

  AllocatedBuffer allocateManaged(size_t bytes);
  AllocatedBuffer allocateDevice(size_t bytes, bool useAllocator = true);
  AllocatedBuffer allocateHostMapped(size_t bytes);
  AllocatedBuffer allocateHost(size_t bytes);
  AllocatedBuffer allocateWriteCombined(size_t bytes);
  AllocatedBuffer allocateDeviceMapped(size_t bytes);

  AllocatedCpuBuffer allocateCpu(size_t bytes);

  AllocatedArray allocateArrayHost(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayHostMapped(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayDevice(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayManaged(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayWriteCombined(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayDeviceMapped(size_t itembytes, size_t numel);

  void createStreamData(std::unique_ptr<StreamData>& ptr);
  StreamData& getStreamData(CUstream stream) {
    auto& ptr = streamData[stream];
    if (!ptr) {
      createStreamData(ptr);
    }
    return *ptr;
  }

  StreamData& getCpuStreamData(uint32_t concurrencyIndex) {
    auto& ptr = cpuStreamData[concurrencyIndex];
    if (!ptr) {
      createStreamData(ptr);
    }
    return *ptr;
  }

  std::atomic_size_t bytesManaged = 0;
  std::atomic_size_t bytesDevice = 0;
  std::atomic_size_t bytesHost = 0;

  void reportBytes() {
    if (currentLogLevel < LOG_VERBOSE) {
      return;
    }
    size_t m = bytesManaged;
    size_t d = bytesDevice;
    size_t h = bytesHost;
    double div = 1024 * 1042 * 1024;
    log.verbose(
        "Bytes managed: %d (%gG)\nBytes device: %d (%gG)\nBytes host: %d (%gG)\n", m, m / div, d, d / div, h, h / div);
  }

  uintptr_t getNextCudaUint32();
  uintptr_t getNextCudaMappedUint32();
};

template<typename T, size_t poolSize = 0x1000>
struct PoolAllocator {
  std::deque<SimpleVector<std::aligned_storage_t<sizeof(T), alignof(T)>>> all;
  size_t i0 = 0;
  size_t i1 = 0;
  PoolAllocator() {
    all.emplace_back();
    all.back().resize(poolSize);
  }
  ~PoolAllocator() {
    reset();
  }
  template<typename... Args>
  T* allocate(Args&&... args) {
    CHECK(i0 < all.size());
    CHECK(i1 < poolSize);
    T* r = (T*)&all[i0][i1];
    ++i1;
    if (i1 == poolSize) {
      i1 = 0;
      ++i0;
      if (i0 == all.size()) {
        all.emplace_back();
        all.back().resize(poolSize);
      }
    }
    return new (r) T(std::forward<Args>(args)...);
  }
  void reset() {
    for (size_t i = 0; i != i0; ++i) {
      for (auto& v : all[i]) {
        ((T*)&v)->~T();
      }
    }
    for (size_t i = 0; i != i1; ++i) {
      ((T*)&all[i0][i])->~T();
    }
    i0 = 0;
    i1 = 0;
  }
};

} // namespace moodist
