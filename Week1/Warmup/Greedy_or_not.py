def score(nums, i, f):

    if i > f:
        return 0
    
    total = sum(nums[i:f+1])

    left = total - score(nums, i+1, f)
    right = total - score(nums, i, f-1)
    return max(left, right)

def main():
    n = int(input())
    play = list(map(int, input().split()))

    total = sum(play)
    score1 = score(play, 0, n-1)
    score2 = total - score1

    if score1 > score2:
        print("Player 1")
    elif score2 > score1:
        print("Player 2")
    else:
        print("Draw")

if __name__ == "__main__":    
    main()
