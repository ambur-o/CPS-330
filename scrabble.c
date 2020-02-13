word = str(input("Enter Word:"))
def letterscore ():
    for x in letterscore:
        if letterscore in 'anorsteuil':
            return 1
        elif letterscore in 'dg':
            return 2
        elif letterscore in 'bcpm':
            return 3
        elif letterscore in 'fhvwz':
            return 4
        elif letterscore in 'k':
            return 5
        elif letterscore in 'qz':
            return 10
        elif letterscore in 'jkx':
            return 8
        else:
            return 0
def wordcount ():
    score = 0
    
    for x in range(len(word)):
        score += letterscore[x]
        return score
print(letterscore())