from flask import Flask, render_template, request, session
from flask_session import Session
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure session to store in the filesystem
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session_data'  # Directory where session files will be saved
app.config['SESSION_PERMANENT'] = False
Session(app)

# Secret key for session management (loaded from environment variable)
app.secret_key = "your_secret_key_here"
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application. Set it as an environment variable.")

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate dataset X and compute dependent variable Y with random noise
    X = np.random.rand(N, 1)  # Feature variable X (N samples)
    Y = beta0 + beta1 * X + mu + np.random.randn(N, 1) * np.sqrt(sigma2)  # Response variable Y

    # Fit a linear regression model to the data (X, Y)
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0][0]  # Model's slope
    intercept = model.intercept_[0]  # Model's intercept

    # Plot the original data points and the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Regression: Y = {slope:.2f}X + {intercept:.2f}")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Simulate multiple datasets and compute slopes and intercepts for hypothesis testing
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim = np.random.rand(N, 1)
        Y_sim = beta0 + beta1 * X_sim + mu + np.sqrt(sigma2) + np.random.randn(N, 1)
        sim_model = LinearRegression().fit(X_sim, Y_sim)
        slopes.append(sim_model.coef_[0][0])
        intercepts.append(sim_model.intercept_[0])

    # Plot histograms of simulated slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, color='blue', alpha=0.7, label='Slopes')
    plt.hist(intercepts, bins=20, color='orange', alpha=0.7, label='Intercepts')
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Simulated Slopes and Intercepts")
    plt.savefig(plot2_path)
    plt.close()

    # Calculate the proportion of slopes and intercepts more extreme than the observed values
    slope_more_extreme = np.mean([abs(s) > abs(slope) for s in slopes])
    intercept_extreme = np.mean([abs(i) > abs(intercept) for i in intercepts])

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and plots based on user input
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store results in session to persist across requests
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # Regenerate the data when the user clicks "Generate"
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data stored in session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Select the statistics (slopes or intercepts) based on user input
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on the chosen test type
    if test_type == "!=":
        p_value = 2 * min(np.mean(simulated_stats >= observed_stat), np.mean(simulated_stats <= observed_stat))
    elif test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:  # test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)

    # Display a fun message if the p-value is extremely small
    fun_message = "Extremely unlikely result!" if p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure(figsize=(10, 5))
    plt.hist(simulated_stats, bins=20, color='gray', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle="dashed", linewidth=2, label="Observed")
    plt.axvline(hypothesized_value, color='blue', linestyle="dotted", linewidth=2, label="Hypothesized")
    plt.legend()
    plt.xlabel("Simulated Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Simulated {parameter.capitalize()} with Observed and Hypothesized Values")
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Select estimates and true parameter based on user input
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard deviation of estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # Calculate confidence interval for the parameter estimate
    confidence_level_fraction = confidence_level / 100
    percentile = 100 * (1 - (1 - confidence_level_fraction) / 2)
    z_critical = np.percentile(np.random.normal(0, 1, size=100000), percentile)

    # Confidence interval using normal approximation
    ci_lower = mean_estimate - z_critical * (std_estimate / np.sqrt(S))
    ci_upper = mean_estimate + z_critical * (std_estimate / np.sqrt(S))

    # Check if the confidence interval includes the true parameter value
    includes_true = (ci_lower <= true_param <= ci_upper)

    # Plot confidence interval visualization
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 5))
    plt.plot(estimates, 'o', color='gray', alpha=0.5, label="Estimates")
    plt.axhline(mean_estimate, color="green", linestyle="--", label="Mean Estimate")
    plt.fill_between(range(len(estimates)), ci_lower, ci_upper, color="blue", alpha=0.2, label="Confidence Interval")
    plt.axhline(true_param, color="red", linestyle="--", label="True Parameter")
    plt.legend()
    plt.title(f"{parameter.capitalize()} Confidence Interval")
    plt.xlabel("Simulation Index")
    plt.ylabel(f"{parameter.capitalize()} Estimate")
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True)
