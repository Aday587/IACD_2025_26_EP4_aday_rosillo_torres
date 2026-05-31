from DataGenerator import generateSamples

data = generateSamples(10)

print(data)


from Adaline import Adaline

adaline = Adaline(0.01)

print("Adaline creado correctamente")