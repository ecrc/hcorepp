/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_SYCL_RUN_CONTEXT_HPP
#define HCOREPP_KERNELS_SYCL_RUN_CONTEXT_HPP

#include <CL/sycl.hpp>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

namespace hcorepp {
    namespace kernels {

        enum class VariableDependency {
            READ,
            READWRITE,
            WRITE
        };

        typedef std::pair<void *, VariableDependency> DepPair;

        /**
         * @brief
         * Class responsible of providing the run context for the SYCL device.
         */
        class RunContext {
        public:
            /**
             * @brief
             * Default Constructor.
             */
            RunContext();

            /**
             * @brief
             * Default copy constructor.
             * @param[in] aContext
             * The context to copy.
             */
            RunContext(const RunContext &aContext);

            /**
             * @brief
             * Constructor taking the device name and selecting the device containing the name.
             * @param[in] aDeviceName
             * The device name to select.
             */
            explicit RunContext(const std::string &aDeviceName);

            void SetDevice(const std::string &aDeviceName);

#if __SYCL_COMPILER_VERSION >= 20221201
            /**
             * @brief
             * Constructor taking a callable method for the queue device selection for the context.
             * @param[in] aSelectorCallable
             * A callable function taking a device and returning the priority of selecting that device,
             * following the SYCL device selector.
             */
            explicit RunContext(int (*aSelectorCallable)(const sycl::device &));
#else

            /**
             * @brief
             * Constructor taking a device selector to use for the queue creation.
             * @param[in] aSelector
             * The device selector to use.
             */
            explicit RunContext(const sycl::device_selector &aSelector);

#endif

            /**
             * @brief
             * Returns sycl scheduler queue
             */
            sycl::queue &GetQueue() const;

            /**
             * @brief
             * Synchronizes the kernel stream.
             */
            void Sync() const;

            /**
             * @brief
             * Synchronizes the events tied to specific addresses.
             * @param[in] aAddresses The list of addresses to wait on its events.
             */
            void Sync(const std::vector<void *> &aAddresses) const;

            /**
             * @brief
             * Add variable to the run context, so it handles its dependencies correctly.
             * @param[in] apAddress
             * The address of the memory variable.
             * @param[in] aBytes
             * The size of the memory.
             */
            void AddVariable(void *apAddress, std::size_t aBytes) const;

            /**
             * @brief
             * Get the last events related to the provided variables.
             * @param[in] aAddresses
             * Vector of addresses to get the associated events with.
             * @return
             * The vector of the last events this variable depends on.
             */
            std::vector<sycl::event> GetVariableEvents(const std::vector<std::pair<void *, VariableDependency>>
                                                       &aAddresses) const;

            /**
             * @brief
             * Sets the event to a given variable.
             * @param[in] aAddresses
             * The addresses to set an event to.
             * @param[in] aEvent
             * The event that's responsible for it.
             */
            void SetVariableEvent(const std::vector<std::pair<void *, VariableDependency>>
                                  &aAddresses, const sycl::event& aEvent) const;

            /**
             * @brief
             * Removes a registered variable from the map.
             * @param[in] apAddress
             * The address of the variable to remove.
             */
            void RemoveVariable(void *apAddress) const;

            /**
             * @brief
             * Prints some information about the queue selection.
             */
            void Print() const;

            /**
             * @brief
             * Forks a child context for safe use inside threads.
             * @return
             * The child context to use.
             */
            RunContext ForkChildContext();

            /**
             * @brief
             * Default destructor.
             */
            ~RunContext();

            /**
             * @brief
             * Check if Context supports OMP Parallelization
             */
            bool SupportsOMP() const {
                return false;
            }

        private:
            /**
             * @brief
             * Get the starting and ending address registered if available for the provided pointer.
             * @param[in] apPointer The pointer to get the base addresses for.
             * @return A pair containing the start and ending addresses if a valid address, or a nullptr
             * if an invalid address.
             */
            std::pair<void *, void *> GetAddress(void *apPointer) const;

        private:
            /// The sycl queue used.
            mutable sycl::queue mQueue;
            /// The variables allocated by the run context, key is ending address, value is the starting address.
            std::shared_ptr<std::map<void *, void *>> mVariables;
            /// Map between the starting address, and the last event responsible for computation inside it.
            std::shared_ptr<std::unordered_map<void *, sycl::event>> mVariableWriteEvents;
            std::shared_ptr<std::unordered_map<void *, std::vector<sycl::event>>> mVariableReadEvents;
        };

    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_SYCL_RUN_CONTEXT_HPP
