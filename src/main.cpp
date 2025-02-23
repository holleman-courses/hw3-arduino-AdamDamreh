#include <Arduino.h>
#include "HW3model.h"
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define SERIAL_BAUD_RATE 115200
#define INPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 7

char in_str_buff[INPUT_BUFFER_SIZE + 1];
int in_buff_idx = 0;

static tflite::MicroErrorReporter micro_error_reporter;
static const tflite::Model* model = tflite::GetModel(model_int8_tflite);
static constexpr int kTensorArenaSize = 8 * 1024;  // Increased to 8 KB
static uint8_t tensor_arena[kTensorArenaSize];

// Updated resolver with 5 ops
static tflite::MicroMutableOpResolver<5> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

bool parse_input(char* buffer, int* output_array, size_t max_numbers) {
  char* token = strtok(buffer, " ");
  size_t count = 0;

  while (token != NULL && count < max_numbers) {
    output_array[count++] = atoi(token);
    token = strtok(NULL, " ");
  }

  return (count == max_numbers);
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  while (!Serial);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  // Register all required ops
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddMul();
  resolver.AddAdd();  // Critical for ADD operator

  interpreter = new tflite::MicroInterpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System ready! Enter 7 numbers:");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      in_str_buff[in_buff_idx] = '\0';
      Serial.print("\nRaw input: ");
      Serial.println(in_str_buff);
      int input_values[INT_ARRAY_SIZE];
      bool valid = parse_input(in_str_buff, input_values, INT_ARRAY_SIZE);

      if (!valid) {
        Serial.println("Error: Need exactly 7 numbers");
      } else {
        for (int i = 0; i < INT_ARRAY_SIZE; i++) {
          input->data.int8[i] = static_cast<int8_t>(input_values[i]);
        }

        unsigned long t_infer_start = micros();
        TfLiteStatus status = interpreter->Invoke();
        unsigned long t_infer_end = micros();

        if (status != kTfLiteOk) {
          Serial.println("Inference failed!");
        } else {
          unsigned long t_print_start = micros();
          Serial.print("Prediction: ");
          Serial.println(output->data.int8[0]);
          unsigned long t_print_end = micros();

          Serial.print("Print time (μs): ");
          Serial.println(t_print_end - t_print_start);
          Serial.print("Inference time (μs): ");
          Serial.println(t_infer_end - t_infer_start);
        }
      }

      in_buff_idx = 0;
      memset(in_str_buff, 0, sizeof(in_str_buff));
      
    } else if (in_buff_idx < INPUT_BUFFER_SIZE) {
      in_str_buff[in_buff_idx++] = c;
    }
  }
}