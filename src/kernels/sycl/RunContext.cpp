/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/kernels/sycl/RunContext.hpp>
#include <utility>

namespace hcorepp {
    namespace kernels {

        RunContext::RunContext(const RunContext &aContext) {
            this->mQueue = aContext.mQueue;
            this->mVariables = aContext.mVariables;
            this->mVariableWriteEvents = aContext.mVariableWriteEvents;
            this->mVariableReadEvents = aContext.mVariableReadEvents;
        }

#if __SYCL_COMPILER_VERSION >= 20221201

        RunContext::RunContext() {
            mQueue = sycl::queue(sycl::default_selector_v);
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
        }

        RunContext::RunContext(int (*aSelectorCallable)(const sycl::device &)) {
            mQueue = sycl::queue(aSelectorCallable);
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
        }

        RunContext::RunContext(const std::string &aDeviceName) {
            if (aDeviceName.empty()) {
                mQueue = sycl::queue(sycl::default_selector_v);
            } else {
                mQueue = sycl::queue([&](const sycl::device &aDevice) -> int {
                    if (aDevice.get_info<sycl::info::device::name>().find(aDeviceName) != std::string::npos) {
                        return 1;
                    }
                    return 0;
                });
            }
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
        }

#else

        RunContext::RunContext() {
            sycl::default_selector selector;
            mQueue = sycl::queue(selector);
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
            mMutex = std::make_shared<std::shared_mutex>();
        }

        RunContext::RunContext(const sycl::device_selector &aSelector) {
            mQueue = sycl::queue(aSelector);
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
            mMutex = std::make_shared<std::shared_mutex>();
        }

        class NameSelector : public sycl::device_selector {
        public:
            explicit NameSelector(std::string aName) : mDeviceName(std::move(aName)) {

            }

            int operator()(const sycl::device &aDevice) const override {
                if (aDevice.get_info<sycl::info::device::name>().find(mDeviceName) != std::string::npos) {
                    return 1;
                }
                return 0;
            }

        private:
            std::string mDeviceName;
        };

        RunContext::RunContext(const std::string &aDeviceName) {
            if (aDeviceName.empty()) {
                sycl::default_selector selector;
                mQueue = sycl::queue(selector);
            } else {
                NameSelector selector(aDeviceName);
                mQueue = sycl::queue(selector);
            }
            mVariables = std::make_shared<std::map<void *, void *>>();
            mVariableWriteEvents = std::make_shared<std::unordered_map<void *, sycl::event>>();
            mVariableReadEvents = std::make_shared<std::unordered_map<void *, std::vector<sycl::event>>>();
            mMutex = std::make_shared<std::shared_mutex>();
        }

#endif

        void RunContext::AddVariable(void *apAddress, std::size_t aBytes) const {
            auto pair = this->GetAddress(apAddress);
            if (pair.first == nullptr) {
                char *end = (char *) apAddress + aBytes;
                (*this->mVariables)[end] = apAddress;
            } else {
                throw std::runtime_error("A variable has been entered twice to the run context");
            }
        }

        std::pair<void *, void *> RunContext::GetAddress(void *apPointer) const {
            std::pair<void *, void *> ret(nullptr, nullptr);
            auto it = this->mVariables->lower_bound((char *) apPointer + 1);
            if (it != this->mVariables->end()) {
                auto end = it->first;
                auto start = it->second;
                if (apPointer >= start && apPointer < end) {
                    ret.first = start;
                    ret.second = end;
                }
            }
            return ret;
        }

        std::vector<sycl::event> RunContext::GetVariableEvents(const std::vector<std::pair<void *, VariableDependency>>
                                                               &aAddresses) const {
            std::vector<sycl::event> ret;
            for (auto &add : aAddresses) {
                auto pair = this->GetAddress(add.first);
                auto dep = add.second;
                if (pair.first != nullptr) {
                    if (dep == VariableDependency::READ) {
                        // If just read request, ensure last write has finished.
                        if (this->mVariableWriteEvents->count(pair.first) > 0) {
                            ret.push_back((*this->mVariableWriteEvents)[pair.first]);
                        }
                    } else {
                        // If write dependency, ensure all previous reads and writes have completed.
                        if (this->mVariableWriteEvents->count(pair.first) > 0) {
                            ret.push_back((*this->mVariableWriteEvents)[pair.first]);
                        }
                        if (this->mVariableReadEvents->count(pair.first) > 0) {
                            auto &vec = (*this->mVariableReadEvents)[pair.first];
                            ret.insert(ret.end(), vec.begin(), vec.end());
                        }
                    }
                }
            }
            return ret;
        }

        void RunContext::Sync(const std::vector<void *> &aAddresses) const {
            std::vector<std::pair<void *, VariableDependency>> deps;
            for (auto add : aAddresses) {
                deps.emplace_back(add, VariableDependency::READWRITE);
            }
            auto evs = this->GetVariableEvents(deps);
            for (auto ev : evs) {
                ev.wait();
            }
        }

        void RunContext::SetVariableEvent(const std::vector<std::pair<void *, VariableDependency>>
                                          &aAddresses, const sycl::event &aEvent) const {
            for (auto &add_pair : aAddresses) {
                auto address = add_pair.first;
                auto dep_type = add_pair.second;
                auto pair = this->GetAddress(address);
                if (pair.first != nullptr) {
                    if (dep_type == VariableDependency::READ) {
                        if (this->mVariableReadEvents->count(pair.first) > 0) {
                            (*this->mVariableReadEvents)[pair.first].push_back(aEvent);
                        } else {
                            (*this->mVariableReadEvents)[pair.first] = {aEvent};
                        }
                    } else {
                        (*this->mVariableWriteEvents)[pair.first] = aEvent;
                        this->mVariableReadEvents->erase(pair.first);
                    }
                }
            }
        }

        void RunContext::RemoveVariable(void *apAddress) const {
            auto pair = this->GetAddress(apAddress);
            if (pair.first != nullptr) {
                this->mVariables->erase(pair.second);
                if (this->mVariableWriteEvents->count(pair.first) > 0) {
                    (*this->mVariableWriteEvents)[pair.first].wait();
                }
                if (this->mVariableReadEvents->count(pair.first) > 0) {
                    auto &deps = (*this->mVariableReadEvents)[pair.first];
                    for (auto ev : deps) {
                        ev.wait();
                    }
                }
                this->mVariableWriteEvents->erase(pair.first);
                this->mVariableReadEvents->erase(pair.first);
            } else {
                throw std::runtime_error("A variable has not been registered to the run context but"
                                         " removal was requested");
            }
        }

        sycl::queue &RunContext::GetQueue() const {
            return mQueue;
        }

        void RunContext::Sync() const {
            mQueue.wait();
        }

        RunContext RunContext::ForkChildContext() {
            RunContext context(*this);
            return context;
        }

        void RunContext::Print() const {
            std::cout << "Device : " << mQueue.get_device().get_info<sycl::info::device::name>() << std::endl;
        }

        RunContext::~RunContext() {
        }

        void RunContext::SetDevice(const std::string &aDeviceName) {
            this->Sync();
            if (aDeviceName.empty()) {
                mQueue = sycl::queue(sycl::default_selector_v);
            } else {
                mQueue = sycl::queue([&](const sycl::device &aDevice) -> int {
                    if (aDevice.get_info<sycl::info::device::name>().find(aDeviceName) != std::string::npos) {
                        return 1;
                    }
                    return 0;
                });
            }
        }

    }//namespace kernels
}//namespace hcorepp

