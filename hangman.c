A = ['h','a','n','g','m','a','n']
L = ['_','_','_','_','_','_','_']
play = True

while play == True:
# Ask the user to guess a letter
    letter = str(input("Guess a letter: "))
# Check to see if that letter is in the Answer
    i = 0
    j = 6
    # for each letter in A do the following
    for currentletter in A:
   
        # If the letter the user guessed is found in the answer,
        # set the underscore in the user's answer to that letter
        if letter == currentletter:
            L[i] = letter
        i = i + 1
    if letter not in A:
        j = j - 1
        print("Bad Guess", j, "chances left")
        
       
    # Display what the player has thus far (L) with a space
    # separating each letter
    print(' '.join(str(n) for n in L))
    
        
            
    # Test to see if the word has been successfully completed,
    # and if so, end the loop
    if A == L:
            play = False

print("GREAT JOB!")