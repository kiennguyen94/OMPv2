################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/OMP.cu \
../src/OMP_alt.cu 

OBJS += \
./src/OMP.o \
./src/OMP_alt.o 

CU_DEPS += \
./src/OMP.d \
./src/OMP_alt.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -DADD_ -I.././include -I/usr/local/magma/include -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -DADD_ -I.././include -I/usr/local/magma/include -G -g -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


