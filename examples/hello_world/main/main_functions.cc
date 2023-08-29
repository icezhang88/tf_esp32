/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "stdlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include "stdlib.h"



#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"

#include "string.h"
//#include "driver/gpio.h"
#include "stdlib.h"


#define TXD_PIN 17
#define RXD_PIN 16
#define UART2 (2)

// Globals, used for compatibility with Arduino-style sketches.
namespace {
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = 50 * 2048;
    uint8_t tensor_arena[kTensorArenaSize];

    static const int RX_BUF_SIZE = 1032;

    uint8_t RX_BUFFER[2056]={0};


}  // namespace
//int sendData(const char* logName, const char* data)
//{
//    const int len = strlen(data);
//    const int txBytes = uart_write_bytes(UART2, data, len);
//    ESP_LOGI(logName, "Wrote %d bytes", txBytes);
//    return txBytes;
//}
//
//
//
//static void rx_task(void *arg)
//{
//    static const char *RX_TASK_TAG = "RX_TASK";
//    esp_log_level_set(RX_TASK_TAG, ESP_LOG_INFO);
//    uint8_t* data = (uint8_t*) malloc(RX_BUF_SIZE+1);
//
//
//    char  sleep_matrix[165]={0};
//    sleep_matrix[0]=0x01;
//
//    sleep_matrix[161]=0xaa;
//    sleep_matrix[162]=0x55;
//    sleep_matrix[163]=0x03;
//    sleep_matrix[164]=0x99;
//
//
//    char  sleep_posture[6]={0};
//    sleep_posture[0]=0x02;
//    sleep_posture[1]=0x00;
//    sleep_posture[2]=0xaa;
//    sleep_posture[3]=0x55;
//    sleep_posture[4]=0x03;
//    sleep_posture[5]=0x99;
//    while (1) {
//        const int rxBytes = uart_read_bytes(UART2, data, RX_BUF_SIZE, 100 / portTICK_RATE_MS);
//
//        if (rxBytes > 0) {
//            data[rxBytes] = 0;
//            //ESP_LOGI(RX_TASK_TAG, "Read %d bytes: '%s'", rxBytes, data);
//            ESP_LOGI(RX_TASK_TAG, "Read %d bytes:  ", rxBytes  );
//            ESP_LOG_BUFFER_HEXDUMP(RX_TASK_TAG, data, rxBytes, ESP_LOG_INFO);
//
//
//
//
//            uint8_t ndata[160]={0};
//            int ndataIndex=0;
//            for (int i = 0; i < 16; ++i) {
//                for (int j = 0; j < 10; ++j) {
//                    ndata[ndataIndex]=data[i * 32 + j];
//                    ndataIndex++;
//                }
//            }
//
//            for (int i = 8; i <12 ; ++i) {
//                for (int j = 0; j < 10; ++j) {
//                    ndata[i * 10 + j]=ndata[(15 - i + 8) * 10 + j];
//                    ndata[(15 - i + 8) * 10 + j]= ndata[i * 10 + j];
//                }
//            }
//
//
//            for (int i = 0; i < 160; ++i) {
//                sleep_matrix[i+1]=ndata[i];
//                input->data.f[i] = ndata[i] / 255.0;
//            }
//
//
//
//
//            TfLiteStatus invoke_status = interpreter->Invoke();
//            if (invoke_status != kTfLiteOk) {
//                MicroPrintf("Invoke failed on x: \n");
//                return;
//            }
//            TfLiteTensor *output = interpreter->output(0);
//
//            char str[250] = {0};
//            char buf[20] = {0};
//            int numElements = output->dims->data[1];
//
//            MicroPrintf("numElements:%d", numElements);
//
//            float max_val = static_cast<float>(output->data.f[0]);
//            int max_index = 0;
//
//            for (int i = 0; i < numElements; i++) {
//                MicroPrintf("%f", static_cast<float>(output->data.f[i]));
//                strcat(str, buf);
//            }
//            for (int i = 1; i < numElements; i++) {
//                if (static_cast<float>(output->data.f[i]) > max_val) {
//                    max_val = static_cast<float>(output->data.f[i]);
//                    max_index = i;
//                }
//            }
//            MicroPrintf("max index: %d", max_index);
//            ESP_LOGI(RX_TASK_TAG,"read bytes conut: %d",rxBytes);
//            sleep_posture[1]=max_index;
//            const int txBytes = uart_write_bytes(UART2, sleep_posture, 6);
//            vTaskDelay(50);
//            MicroPrintf("txBytes:%d", txBytes);
//            vTaskDelay(50);
//            uart_write_bytes(UART2, sleep_matrix, 165);
//
//        }
//    }
//    free(data);
//}
// The name of this function is important for Arduino compatibility.
void setup() {


    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_model);
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }
    MicroPrintf("model load success");
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);


    output = interpreter->output(0);

//    const uart_config_t uart_config = {
//            .baud_rate = 1000000,
//            .data_bits = UART_DATA_8_BITS,
//            .parity = UART_PARITY_DISABLE,
//            .stop_bits = UART_STOP_BITS_1,
//            .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
//            .source_clk = UART_SCLK_APB,
//    };
//
//    // We won't use a buffer for sending data.
//    uart_driver_install(UART2, RX_BUF_SIZE * 2, 0, 0, NULL, 0);
//    uart_param_config(UART2, &uart_config);
//    uart_set_pin(UART2, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);


  // init();
   // xTaskCreate(rx_task, "uart_rx_task", 1024*4, NULL, configMAX_PRIORITIES, NULL);
   // xTaskCreate(tx_task, "uart_tx_task", 1024*2, NULL, configMAX_PRIORITIES-1, NULL);



}

// The name of this function is important for Arduino compatibility.
void loop() {
    // Calculate an x value to feed into the model. We compare the current
    // inference_count to the number of inferences per cycle to determine
    // our position within the range of possible x values the model was
    // trained on, and use this to calculate a value.
    
//    float arr[160] = {
//            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 18, 9, 19, 18, 8, 7, 12, 16, 11, 21, 45, 41, 33, 11, 17, 14, 32, 29,
//            31, 33, 29, 38, 32, 15, 3, 22, 8, 21, 19, 27, 14, 20, 21, 15, 17, 1, 0, 3, 5, 5, 9, 0, 9, 11, 17, 0, 0, 0,
//            1, 7, 1, 1, 11, 6, 12, 5, 0, 0, 0, 0, 0, 0, 1, 11, 18, 3, 0, 0, 0, 0, 0, 0, 6, 9, 26, 11, 0, 0, 0, 0, 0, 1,
//            9, 57, 26, 1, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0
//    };
    float arr[2048]={100};
    for (int i = 0; i < 2048; i++) {
        input->data.f[i] = 38    / 255.0;
    }


    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed on x: \n");
        return;
    }
    TfLiteTensor *output = interpreter->output(0);
//    const  float* output_data = output->data.f;
//    int len = sizeof(output_data) / sizeof(float);
    char str[250] = {0};
    char buf[20] = {0};
    int numElements = output->dims->data[1];

    MicroPrintf("numElements:%d", numElements);
    for (int i = 0; i < 6; i++) {
     //   MicroPrintf("%f", static_cast<float>(output->data.f[i]));
//        MicroPrintf(buf, "%e,", static_cast<float>(output->data.f[i]));
//        sprintf(buf, "%e,", static_cast<float>(output->data.f[i]));
//        strcat(str, buf);
    }

    MicroPrintf("-----------------------");


    float output_data[4];
    for (int i = 0; i < 4; ++i) {
        output_data[i] = output->data.f[i];
    }
    MicroPrintf("0 : %f  1 : %f  2 : %f 3 ：%f\n",output_data[0],output_data[1],output_data[2],output_data[3]);

}
