from processamento.DataTable import DataTable

class Main:
    def __init__(self) -> None:
        self.dataTable = DataTable()
    
    def main(self):
        self.dataTable.writeDataFrame()
        

if __name__ == "__main__":
    main = Main()
    main.main()