def obtain_results(file_path:str) -> list[list[int]]:
    sudoku = []
    file = open(file_path, "r")

    for i,line in enumerate(file):
        if i > 1:
            row = line.split(" ")
            sudoku.append(row)
    
    return sudoku
