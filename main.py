from processamento.DataTable import DataTable

class Main:
    def __init__(self) -> None:
        self.dataTable = DataTable()
    
    def main(self):
        self.dataTable.writeDataTableIntoFile()
        

if __name__ == "__main__":
    main = Main()
    main.main()