#######################################################################################################
#                                                                $ python3 programm to convert                                                                        #
#                                                                $ String into all Other                                                                                        #
#######################################################################################################
import time
## no. 1
def str2bin():
    binary_conversion=[]
    String=input('Enter Statement:  ')
    for i in String:
        ascii_value=ord(i)
        binary_value=bin(ascii_value)
        binary_conversion.append(binary_value[2:])
        output=(''.join(binary_conversion))
    print("binary of total string: ",output)


#no. 2
def str2int():
    integer_value=[]
    ascii_value=[]
    key=''
    string_key=[]
    count=0
    String=input("Enter Statement: ")
    for i in String:
        ascii_value=ord(i)
        integer_value.append(ascii_value)
        key=i
        string_key.append(key)
        count=count+1
    print('length of string is:',count)
    for x in range(0,count):
        print('charcte is: ',string_key[x],'integer is: ',integer_value[x])

# no. 3
def str2chrascii():
    count_in_loop=0
    char_key=[]
    char_value=[]
    char_key_temp=''
    ascii_value=0
    String=input("Eneter Statement: ")
    for x in String:
        char_key_temp=x
        ascii_value=ord(x)
        char_key.append(char_key_temp)
        char_value.append(ascii_value)
        count_in_loop=count_in_loop+1
    for y in range(0,count_in_loop):
        print("char is: ",char_key[y],'ascii value is: ',char_value[y])

# no. 4
def str2chrbin():
    count=0
    char_key=[]
    char_value=[]
    key=''
    ascii_value=0
    value=0
    String=input("Enter the String: ")
    for x in String:
        ascii_value=ord(x)
        value=bin(ascii_value)
        char_value.append(value)
        key=x
        char_key.append(key)
        count=count+1
    print("length of your character is: ",count)
    for y in range(0,count):
        print("char is: ",char_key[y],"binary value is: ",char_value[y])



print('###########################################################################################')
print('#               Enter 1                  for  String to Binary                            #')
print('#               Enter 2                  for  String to integer                           #')
print('#               Enter 3                  for  char and them Ascii code                    #')
print('#               Enter 4                  for  char and them binary code                   #')
print('#                      |||||||||||||||||||||||||||||||||||||||||||||||||                  #')
print('#               Enter 9                  for "exit" or "stop                            #')
print('###########################################################################################')


number=int(input("Enter the number: "))
if number == 1:
    str2bin()
    time.sleep(3)
elif number ==2:
    str2int()
    time.sleep(3)
elif number == 3:
    str2chrascii()
    time.sleep(3)
elif number == 4:
    str2chrbin()
    time.sleep(3)
elif number == 9:
    exit()


##    
##def str2bin():
##    string=input("enter the value:")
##    a=''.join(format(ord(x),'b') for x in (string))
##    print(a)
