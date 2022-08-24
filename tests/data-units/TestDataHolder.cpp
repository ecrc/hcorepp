
#include <libraries/catch/catch.hpp>

#include <hcorepp/data-units/DataHolder.hpp>

using namespace std;
using namespace hcorepp::dataunits;

template<typename T>
void TEST_DATA_HOLDER() {
    SECTION("Data Holder creation test") {
        int n_rows = 5;
        int n_cols = 5;

        T *data_array = new T[n_rows * n_cols];

        for (int i = 0; i < n_rows * n_cols; i++) {
            data_array[i] = i;
        }

        DataHolder<T> data_holder = DataHolder<T>(n_rows, n_cols, n_rows, data_array);

        REQUIRE(data_holder.GetNumOfRows() == n_rows);
        REQUIRE(data_holder.GetNumOfCols() == n_cols);
        auto data = data_holder.GetData();
        for (int i = 0; i < n_rows * n_cols; i++) {
            REQUIRE(data[i] == i);
        }

        delete[] data_array;
    }

    SECTION("Data Holder Resize test") {
        int n_rows = 5;
        int n_cols = 5;

        T *data_array = new T[n_rows * n_cols];

        for (int i = 0; i < n_rows * n_cols; i++) {
            data_array[i] = i;
        }

        DataHolder<T> data_holder = DataHolder<T>(n_rows, n_cols, n_rows, data_array);

        int new_rows = 6;
        int new_cols = 6;
        data_holder.Resize(new_rows, new_cols, n_rows);

        T *new_data_array = new T[new_rows * new_cols];

        for (int i = 0; i < new_rows * new_cols; i++) {
            new_data_array[i] = i * 2;
        }

        data_holder.CopyDataArray(0, new_data_array, new_rows * new_cols);

        REQUIRE(data_holder.GetNumOfRows() == new_rows);
        REQUIRE(data_holder.GetNumOfCols() == new_cols);

        auto data = data_holder.GetData();
        for (int i = 0; i < new_rows * new_cols; i++) {
            REQUIRE(data[i] == i * 2);
        }

        delete[] data_array;
        delete[] new_data_array;

    }
}

TEMPLATE_TEST_CASE("DataHolderTest", "[DATAHOLDER]", float, double) {
    TEST_DATA_HOLDER<TestType>();
}
