#include <EdgePaddingLib.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <chrono>

namespace py = pybind11;

py::array_t<uint8_t> edge_padding_uint8_custom_mask(py::array_t<uint8_t> input, py::array_t<uint8_t> mask) {

    constexpr int INPUT_CHANNELS = 4;

    py::buffer_info i_buf_info = input.request();
    py::buffer_info m_buf_info = mask.request();

    if (i_buf_info.ndim != 3 || i_buf_info.shape[2] != INPUT_CHANNELS) {
        throw (std::runtime_error("Input shape should be [m, n, 4] !"));
    }

    int INPUT_WIDTH = i_buf_info.shape[1];
    int INPUT_HEIGHT = i_buf_info.shape[0];

    if (m_buf_info.ndim != 2 || m_buf_info.shape[0] != INPUT_HEIGHT || m_buf_info.shape[1] != INPUT_WIDTH) {
        throw (std::runtime_error("Mask shape should be [m, n] !"));
    }
    
    py::array_t<uint8_t> output(i_buf_info.shape);
    py::buffer_info o_buf_info = output.request();

    EdgePadding::FillZeroPixels((uchar4*)i_buf_info.ptr, (uchar4*)o_buf_info.ptr, INPUT_WIDTH, INPUT_HEIGHT, (uint8_t*)m_buf_info.ptr);

    return output;
}

py::array_t<uint8_t> edge_padding_uint8(py::array_t<uint8_t> input) {

    constexpr int INPUT_CHANNELS = 4;

    py::buffer_info i_buf_info = input.request();

    if (i_buf_info.ndim != 3 || i_buf_info.shape[2] != INPUT_CHANNELS) {
        throw (std::runtime_error("Input shape should be [m, n, 4] !"));
    }

    int INPUT_WIDTH = i_buf_info.shape[1];
    int INPUT_HEIGHT = i_buf_info.shape[0];

    py::array_t<uint8_t> output(i_buf_info.shape);
    py::buffer_info o_buf_info = output.request();

    EdgePadding::FillZeroPixels((uchar4*)i_buf_info.ptr, (uchar4*)o_buf_info.ptr, INPUT_WIDTH, INPUT_HEIGHT, nullptr);

    return output;
}

PYBIND11_MODULE(PyEdgePadding, m) {
    m.def("edge_padding_uint8_custom_mask", &edge_padding_uint8_custom_mask, "Texture Image Edge Padding with Custom Mask");
    m.def("edge_padding_uint8", &edge_padding_uint8, "Texture Image Edge Padding");
}