import random


def Guessing_Game():
    random_integer = random.randint(1, 9)  # Generate integer from 1-9 randomly
    # print(random_integer)
    count_guess = 0  # Count the number of guessing
    while True:
        guess = input('Guess a integer (1-10): ')
        if guess == 'exit':
            print('You guess {} times.'.format(count_guess))
            break
        count_guess += 1
        if int(guess) > random_integer:
            print('Higher')
        elif int(guess) < random_integer:
            print('Lower')
        else:
            print('Exactly right')


Guessing_Game()


def Check_Primality():
    random_integer = random.randint(1, 1000)  # Generate a integer range from 1-1000
    for i in range(2, int(random_integer ** 0.5) + 1):
        ''' Checking whether integers can be divided by numbers from 2 to square root of the integer '''
        if random_integer % i == 0:
            print('{} isn\'train_t a prime number.'.format(random_integer))
            return
    print('{} is a prime number.'.format(random_integer))
    return


Check_Primality()


def Remove_Duplicates(input: list) -> list:
    output = []
    [output.append(i) for i in input if not i in output]
    return output


print(Remove_Duplicates([1,3,2,3,4,5,1,5,2]))


def Max_of_Three(num1, num2, num3):
    if num1 > num2:
        max_num = num1
    else:
        max_num = num2

    if max_num < num3:
        max_num = num3
    #print("Max of three：{}".format(max_num))
    return max_num


print(Max_of_Three(5, 8, 20))
