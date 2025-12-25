import yfinance as yf

def norm(ticker):
    df = yf.download(f'{ticker}', start='2022-01-01', end='2025-01-01')
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.to_csv(f'./Stock Data/{ticker}.csv')

def test(ticker):
    df = yf.download(f'{ticker}', start='2025-01-01', end='2025-12-10')
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.to_csv(f'./Stock Data/{ticker}TEST.csv')

while True:
    ticker = input('Enter ticker symbol: ').upper()
    if ticker == 'EXIT':
        break

    type = input('Enter stock type: ')

    if type == 'norm':
        norm(ticker)

    elif type == 'test':
        test(ticker)