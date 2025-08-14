from flask import Flask, render_template, request
from predictor import fetch_data, add_features, train_and_predict, plot_results

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = {}
    error = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper().strip()
        model_type = request.form.get("model", "ridge")
        period = request.form.get("period", "6mo")

        try:
            df = fetch_data(ticker, period=period)

            #Extra safety check
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}.")

            df = add_features(df)
            results = train_and_predict(df, ticker, model_type=model_type)
            plot_path = plot_results(df, results["pred_price"], ticker)

            return render_template(
                "index.html",
                ticker=ticker,
                pred_price=round(results["pred_price"], 2),
                pred_return=round(results["pred_return"] * 100, 2),
                last_close=round(results["last_close"], 2),
                mse=round(results["mse"], 5),
                r2=round(results["r2"], 3),
                trade_signal=results["trade_signal"],
                plot_path=plot_path,
                model_selected=model_type,
                period_selected=period,
                error=None
            )
        except Exception as e:
            error = str(e)

    return render_template("index.html", error=error)

if __name__ == "__main__":
    app.run(debug=True)

