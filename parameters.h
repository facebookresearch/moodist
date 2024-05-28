#pragma once

#include "cputhread.h"
#include "group.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>

namespace moodist {

template<size_t uid, typename TT, size_t numOptionsT>
struct Parameter {
  using T = TT;
  static constexpr size_t numOptions = numOptionsT;
  std::string name;
  T defaultValue;
  std::array<T, numOptions> options;

  Parameter(std::string name, T defaultValue, std::array<T, numOptions> options) {
    this->name = name;
    this->defaultValue = defaultValue;
    this->options = options;
  }
};

template<typename... Parameter>
struct SuperParameter {
  std::tuple<Parameter&...> params;
  using T = size_t;
  static constexpr size_t numOptions = (Parameter::numOptions * ...);
  std::string name;
  T defaultValue = 0;
  std::array<T, numOptions> options;
  SuperParameter(std::string name, Parameter&... p) : name(name), params(p...) {
    for (size_t i = 0; i != numOptions; ++i) {
      options[i] = i;
    }
  }

  template<typename P, size_t index = sizeof...(Parameter) - 1>
  typename P::T get(P& p, size_t value) {
    if constexpr (std::is_same_v<P, std::decay_t<decltype(std::get<index>(params))>>) {
      return std::get<index>(params).options[value % std::decay_t<decltype(std::get<index>(params))>::numOptions];
    } else {
      value /= std::decay_t<decltype(std::get<index>(params))>::numOptions;
      return get<P, index - 1>(p, value);
    }
  }
};

template<typename T>
struct is_super_t {
  static const bool value = false;
};

template<typename... P>
struct is_super_t<SuperParameter<P...>> {
  static const bool value = true;
};

template<typename T>
constexpr bool is_super = is_super_t<T>::value;

// template<typename T>
// struct ParameterInstance {
//   T currentValue;
//   uint32_t stepValue = 0;

//   T operator()(uint32_t stepValue) const {
//     CHECK(stepValue == this->stepValue);
//     return currentValue;
//   }
// };

inline Parameter<__LINE__, size_t, 3> paramNumDevices("num_devices", 2, {1, 2, 4});
inline Parameter<__LINE__, size_t, 4> paramNumChunks("num_chunks", 2, {1, 2, 4, 8});
inline Parameter<__LINE__, size_t, 3> paramNumParallel("num_paralllel", 2, {1, 2, 4});

inline Parameter<__LINE__, size_t, 4> paramAlgo("algo", 0, {0, 1, 2, 3});

// inline Parameter<__LINE__, size_t, 1> paramNumDevices("num_devices", 2, {2});
// inline Parameter<__LINE__, size_t, 4> paramNumChunks("num_chunks", 2, {1, 2, 4, 8});
// inline Parameter<__LINE__, size_t, 1> paramNumParallel("num_paralllel", 2, {2});

inline SuperParameter allGatherSuper("all_gather_params", paramNumDevices, paramNumChunks, paramNumParallel, paramAlgo);

template<typename F, typename Tuple>
void forEach(F&& f, Tuple& tuple) {
  std::apply([&](auto&... x) { (f(x), ...); }, tuple);
}

template<typename... P>
struct ParametersOptimizer {
  std::tuple<P*...> params;
  // std::tuple<ParameterInstance<P>...> instances;
  ParametersOptimizer(P&... p) : params(&p...) {
    forEach(
        [&](auto& p) {
          forEach(
              [&](auto& p2) {
                if constexpr (std::is_same_v<decltype(p), decltype(p2)>) {
                  if (&p != &p2) {
                    log.error("Parameter types not unique: %s & %s\n", p->name, p2->name);
                  }
                  CHECK(&p == &p2);
                }
              },
              params);
        },
        params);
  }

  struct Values {
    std::tuple<typename P::T...> values{};
    std::tuple<typename P::T...> newValues{};
    // std::tuple<std::array<float, P::numOptions>...> weights{};
    bool syncing = false;
    std::atomic_uint32_t done = 0;

    template<typename PT>
    struct Data {
      PT* p = nullptr;
      size_t index = 0;
      size_t newIndex = 0;
      size_t bestIndex = 0;
      std::array<float, PT::numOptions> weights{};
      std::array<float, PT::numOptions> times{};
    };

    std::tuple<Data<P>...> data;

    // std::tuple<std::array<float, P::numOptions>...> times{};

    size_t counter = 0;

    std::chrono::steady_clock::time_point syncTime;

    std::minstd_rand rng{(std::minstd_rand::result_type)std::chrono::steady_clock::now().time_since_epoch().count()};

    template<size_t index = sizeof...(P) - 1>
    void init(std::tuple<P*...>& params) {
      std::get<index>(values) = std::get<index>(params)->defaultValue;
      auto& options = std::get<index>(params)->options;
      auto i = std::find(options.begin(), options.end(), std::get<index>(params)->defaultValue);
      CHECK(i != options.end());
      std::get<index>(data).index = i - options.begin();
      std::get<index>(data).bestIndex = i - options.begin();
      std::get<index>(data).p = std::get<index>(params);
      if constexpr (index != 0) {
        init<index - 1>(params);
      }
    }

    // template<size_t index = sizeof...(P) - 1>
    // void set(std::tuple<ParameterInstance<P>...>& instances) {
    //   std::get<index>(instances).currentValue = std::get<index>(values);
    //   if constexpr (index != 0) {
    //     init<index - 1>(instances);
    //   }
    // }

    template<size_t index = sizeof...(P) - 1>
    void setNewValues() {
      std::get<index>(newValues) = std::get<index>(data).p->options[std::get<index>(data).newIndex];
      if constexpr (index != 0) {
        setNewValues<index - 1>();
      }
    }

    template<size_t N>
    std::array<float, N> log_softmax(const std::array<float, N>& input) {
      float max = input[0];
      for (auto& v : input) {
        max = std::max(max, v);
      }
      float sum = 0.0f;
      for (auto& v : input) {
        sum += std::exp(v - max);
      }
      float logSum = std::log(sum);
      std::array<float, N> r;
      for (size_t i = 0; i != N; ++i) {
        r[i] = input[i] - max - logSum;
      }
      return r;
    }

    template<size_t N>
    std::array<float, N> softmax(const std::array<float, N>& input) {
      float max = input[0];
      for (auto& v : input) {
        max = std::max(max, v);
      }
      std::array<float, N> r;
      float sum = 0.0f;
      for (size_t i = 0; i != N; ++i) {
        r[i] = std::exp(input[i] - max);
        sum += r[i];
      }
      for (auto& v : r) {
        v /= sum;
      }
      return r;
    }

    template<typename ProcessGroup>
    void sync(ProcessGroup* pg) {
      auto now = std::chrono::steady_clock::now();
      if (syncing) {
        if (pg->rank == 0) {
          auto elapsed = now - syncTime;

          float t = seconds(elapsed);
          forEach(
              [&](auto& v) {
                if (v.times[v.index] == 0) {
                  v.times[v.index] = t;
                } else {
                  v.times[v.index] = v.times[v.index] * 0.99f + t * 0.01f;
                }
                // if (v.times[v.index] == 0) {
                //   v.times[v.index] = t;
                // } else {
                //   v.times[v.index] = std::min(v.times[v.index], t);
                // }
                // v.times[v.index] = t;

                for (size_t i = 0; i != v.times.size(); ++i) {
                  if (v.times[i] < v.times[v.bestIndex]) {
                    v.bestIndex = i;
                  }
                }

                // float reward = -1.0f;
                // if (v.times[v.index] <= v.times[v.bestIndex]) {
                //   reward = 1.0f;
                // }
                float reward = v.times[v.bestIndex] / v.times[v.index];

                log.info("%s times: [%s]\n", v.p->name, fmt::to_string(fmt::join(v.times, ", ")));
                log.info("%s best index: %d\n", v.p->name, v.bestIndex);

                log.info("%s update index %d reward %g\n", v.p->name, v.index, reward);

                log.info("%s pre weights [%s]\n", v.p->name, fmt::to_string(fmt::join(v.weights, ", ")));

                auto sf = softmax(v.weights);

                log.info("%s softmax [%s]\n", v.p->name, fmt::to_string(fmt::join(sf, ", ")));

                constexpr float lr = 0.1f;
                constexpr float maxLogValue = std::log(1000 * sf.size());
                constexpr float decay = maxLogValue / (maxLogValue + lr);

                // for (size_t i = 0; i != v.weights.size(); ++i) {
                //   float prob = sf[i];
                //   // if (i == v.index) {
                //   //   // v.weights[i] += lr * reward;
                //   //   v.weights[i] = v.weights[i] * 0.99f + (reward * 0.01f * maxLogValue);
                //   // }
                //   if (i == v.index) {
                //     v.weights[i] += (1 - prob) * reward * lr;
                //   } else {
                //     v.weights[i] -= prob * reward * lr;
                //   }
                //   // v.weights[i] *= decay;
                // }

                // v.weights[v.index] = v.weights[v.index] * 0.98f + reward * 0.02f;

                log.info("%s post weights [%s]\n", v.p->name, fmt::to_string(fmt::join(v.weights, ", ")));

                // log("%s weight[%d] -> %f\n", v.p->name, v.index, v.weights[v.index]);
              },
              data);
        }

        while (!done) {
          log.info("waiting for previous sync! D:\n");
          futexWait(&done, 1, std::chrono::seconds(1));
        }
        syncing = false;
        values = newValues;

        std::string str;
        forEach(
            [&](auto& v) {
              if (!str.empty()) {
                str += " ";
              }
              str += std::to_string(v);
            },
            values);
        log.info("got new values: [%s]\n", str);

        forEach([&](auto& v) { v.index = v.newIndex; }, data);
      }

      if (pg->rank == 0) {
        forEach(
            [&](auto& v) {
              std::array<float, v.weights.size()> probs{};
              float sum = 0.0f;
              for (size_t i = 0; i != probs.size(); ++i) {
                probs[i] = std::exp(v.weights[i]);
                sum += probs[i];
              }
              float value = std::uniform_real_distribution<float>(0.0f, sum)(rng);
              log.info(
                  "%s: probs [%s] sum %g value %g\n", v.p->name, fmt::to_string(fmt::join(probs, ", ")), sum, value);
              size_t i = 0;
              while (i != probs.size() && value > probs[i]) {
                value -= probs[i];
                ++i;
              }
              log.info("%s sampled index %d\n", v.p->name, i);
              v.newIndex = i;
            },
            data);
        setNewValues();

        std::string str;
        forEach(
            [&](auto& v) {
              if (!str.empty()) {
                str += " ";
              }
              str += std::to_string(v);
            },
            newValues);
        log.info("sync new values: [%s]\n", str);
      }
      syncTime = now;
      syncing = true;
      uint32_t stepValue = pg->getNextStepValue();
      uint32_t concurrencyIndex =
          std::exchange(pg->nextConcurrencyIndex, (pg->nextConcurrencyIndex + 1) % Group::maxConcurrency);
      StreamData& sd = pg->group->getStreamData(nullptr);
      QueueEntryBroadcastCpu* e = pg->group->cpuThread->freelistBroadcastCpu.pop();
      e->task = taskBroadcastCpu;
      e->stepValue = stepValue;
      e->concurrencyIndex = concurrencyIndex;
      e->sd = &sd;
      e->tensorAddress = (uintptr_t)(void*)&newValues;
      e->bytes = sizeof(newValues);
      e->sourceRank = 0;
      e->cpuDone = &done;
      pg->group->cpuThread->enqueue(e);
    }

    template<typename Parameter, size_t index = sizeof...(P) - 1>
    typename Parameter::T get(Parameter& p) const {
      if constexpr (is_super<std::decay_t<decltype(*std::get<index>(params))>>) {
        return std::get<index>(data).p->get(p, std::get<index>(values));
      } else if constexpr (std::is_same_v<Parameter, std::decay_t<decltype(*std::get<index>(params))>>) {
        return std::get<index>(values);
      } else {
        return get<Parameter, index - 1>(p);
      }
    }

    template<typename Parameter>
    typename Parameter::T operator()(Parameter& p) const {
      return get(p);
    }
  };

  HashMap<size_t, std::unique_ptr<Values>> valuesMap;

  template<typename ProcessGroup>
  Values& operator()(ProcessGroup* pg, size_t bytes) {
    // auto i = valuesMap.try_emplace(bytes);
    // Values& values = i.first->second;
    auto& valuesPtr = valuesMap[bytes];
    if (!valuesPtr) {
      valuesPtr = std::make_unique<Values>();
      valuesPtr->init(params);
    }
    Values& values = *valuesPtr;
    // std::apply([&](auto& v) { v.stepValue = stepValue; }, params);

    ++values.counter;
    if (values.counter == 8) {
      values.counter = 0;
      values.sync(pg);
    }

    // values.set(instances);

    return values;
  }
};

} // namespace moodist
