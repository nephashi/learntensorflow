training_set = "123"

def my_input_function(para):
    print(para)

def my_input_function_training_set():
  return my_input_function

func = my_input_function_training_set()

func(training_set)