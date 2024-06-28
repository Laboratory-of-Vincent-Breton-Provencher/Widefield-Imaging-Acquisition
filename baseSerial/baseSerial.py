import serial
import time
import serial.tools
import serial.tools.list_ports



# Connection to the Arduino
def getPorts():
    ports = serial.tools.list_ports.comports()
    return ports

def findArduino(portsFound):
    commPort = None
    numConnection = len(portsFound)

    for i in range(0, numConnection):
        port = portsFound[i]
        strPort = str(port)

        if 'Arduino' in strPort:
            splitPort = strPort.split(' ')
            commPort = splitPort[0]

    return commPort

foundPorts = getPorts()
connectPort = findArduino(foundPorts)

if connectPort != None:
    ser = serial.Serial(connectPort, baudrate=9600, timeout=1)
    print('connected to', connectPort)

else:
    print('Connection issue: No Arduino found')


# Creation of empty data file
# dataFile = open("WFdata.csv", 'w')

def getValues():
    arduinoData = ser.readline().decode('ascii').split('\r\n')
    return arduinoData[0]



# Program loop
while True:
    userInput = input('Action:')
    # print(userInput)
    ser.write(b'userInput')
    print(getValues())
    # data = getValues()
    # dataFile.write(data + '\n')