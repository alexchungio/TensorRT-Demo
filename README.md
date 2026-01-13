# TensorRT-Demo
TensorRT  inference library based on C++





### FP16 precision 

make sure the input_data and output_data format to be fp16
the output_data abtained by model, witch format is float16. whereas the C++ have no this format ,so we need to custom the function to convert float16 to float32 before to show
