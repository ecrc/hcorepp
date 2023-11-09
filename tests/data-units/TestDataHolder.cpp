/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <catch2/catch_all.hpp>
#include <iostream>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/kernels/ContextManager.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <cstring>

using namespace std;
using namespace hcorepp::dataunits;

template<typename T>
void TEST_DATA_HOLDER() {
    hcorepp::kernels::RunContext& context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    SECTION("Data Holder creation test") {
        int n_rows = 5;
        int n_cols = 5;

        T *data_array = new T[n_rows * n_cols];

        for (int i = 0; i < n_rows * n_cols; i++) {
            data_array[i] = i;
        }

        DataHolder<T> data_holder(n_rows, n_cols, n_rows, data_array, context);

        REQUIRE(data_holder.GetNumOfRows() == n_rows);
        REQUIRE(data_holder.GetNumOfCols() == n_cols);
        auto data = data_holder.GetData();
        T *host_data_array = new T[n_rows * n_cols];

        hcorepp::memory::Memcpy<T>(host_data_array, data, n_rows * n_cols, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        for (int i = 0; i < n_rows * n_cols; i++) {
            REQUIRE(abs(host_data_array[i] - i) < std::numeric_limits<T>::epsilon());
        }

        delete[] data_array;
        delete[] host_data_array;
    }

    SECTION("Data Holder Resize test") {
        int n_rows = 5;
        int n_cols = 5;

        T *data_array = new T[n_rows * n_cols];

        for (int i = 0; i < n_rows * n_cols; i++) {
            data_array[i] = i;
        }

        DataHolder<T> data_holder(n_rows, n_cols, n_rows, data_array, context);

        int new_rows = 6;
        int new_cols = 6;
        data_holder.Resize(new_rows, new_cols, n_rows);

        T *host_data_array = new T[new_rows * new_cols];

        hcorepp::memory::Memcpy<T>(host_data_array, data_holder.GetData(), new_rows * new_cols,
                                   context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        for (int i = 0; i < new_rows * new_cols; i++) {
            REQUIRE(abs(host_data_array[i]) < std::numeric_limits<T>::epsilon());
        }

        T *new_data_array = new T[new_rows * new_cols];

        for (int i = 0; i < new_rows * new_cols; i++) {
            new_data_array[i] = i * 2;
        }

        hcorepp::memory::Memcpy<T>(&data_holder.GetData()[0], new_data_array, new_rows * new_cols,
                                   context, hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
        context.Sync();
        REQUIRE(data_holder.GetNumOfRows() == new_rows);
        REQUIRE(data_holder.GetNumOfCols() == new_cols);

        hcorepp::memory::Memcpy<T>(host_data_array, data_holder.GetData(), new_rows * new_cols,
                                   context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < new_rows * new_cols; i++) {
            REQUIRE(abs(host_data_array[i] - i * 2) < std::numeric_limits<T>::epsilon());
        }

        delete[] data_array;
        delete[] new_data_array;
        delete[] host_data_array;

    }

}

TEMPLATE_TEST_CASE("DataHolderTest", "[DATAHOLDER]", float, double) {
    TEST_DATA_HOLDER<TestType>();
    hcorepp::kernels::ContextManager::DestroyInstance();
}
