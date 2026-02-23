from utils import *

def main():
    portfolio_df = parse_xls("data/2026S Portfolio Holdings and Analytics.xlsx")
    portfolio_df.to_csv("data/output.csv")

if __name__ == "__main__":
    main()