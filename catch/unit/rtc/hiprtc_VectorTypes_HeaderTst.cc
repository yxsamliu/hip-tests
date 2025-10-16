/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
* @addtogroup hiprtc_VectorTypes_HeaderTst hiprtc_VectorTypes_HeaderTst
* @{
* @ingroup hiprtcHeaders
* `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
*                                  int numOptions,
*                                  const char** options);` -
* These test cases are target including various header file in kernel
* string and compile using the api mentioned above.
*/

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip_test_common.hh>

static constexpr auto vectorTypes_string = R"(
extern "C" __global__
void vectorTypes(char3* out) {
    char3 ch31 = char3(3,  3,  9);
    char3 divisor = char3(2,  3, -3);
    char3 result  = ch31 / divisor;
    out[0] = result;
}
)";

TEST_CASE("Unit_Rtc_VectorTypes_header") {
  std::string kernel_name = "vectorTypes";
  const char* kername = kernel_name.c_str();

#ifdef __HIP_PLATFORM_AMD__
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string complete_CO = "--gpu-architecture=" + std::string(prop.gcnArchName);
#else
  std::string complete_CO = "--fmad=false";
#endif
  const char* compiler_option = complete_CO.c_str();

  // Host/device buffer for exactly one char3
  char3 result_h;
  char3* result_d;
  HIP_CHECK(hipMalloc(&result_d, sizeof(char3)));
  HIP_CHECK(hipMemset(result_d, 0, sizeof(char3)));

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, vectorTypes_string, kername, 0, nullptr, nullptr));
  hiprtcResult compileResult = hiprtcCompileProgram(prog, 1, &compiler_option);
  REQUIRE(compileResult == HIPRTC_SUCCESS);

  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  hipModule_t   module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));

  size_t paramBufferSize = sizeof(result_d);
  void*  kernelArgs[]    = { &result_d };
  void*  launchParams[]  = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
    HIP_LAUNCH_PARAM_END
  };
  HIP_CHECK(hipModuleLaunchKernel(function, 1,1,1, 1,1,1, 0, 0, nullptr, launchParams));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(&result_h, result_d, sizeof(char3), hipMemcpyDeviceToHost));
  REQUIRE(result_h.x ==  1);
  REQUIRE(result_h.y ==  1);
  REQUIRE(result_h.z == -3);

  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipFree(result_d));
}
