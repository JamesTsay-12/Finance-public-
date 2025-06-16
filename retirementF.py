import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import json
import os

# Default parameters
num_simulations = 1000
fat_tail_df = 3

# Default returns and volatilities
normal_returns = {'stocks': 0.085, 'bonds': 0.04, 'cash': 0.03, 'spec': 0.12, 'gold': 0.04, 'reits': 0.06}
historical_avg_returns = {'stocks': 0.095, 'bonds': 0.025, 'cash': 0.0273, 'spec': 0.09, 'gold': 0.02, 'reits': 0.06}
normal_volatility = {'stocks': 0.15, 'bonds': 0.05, 'cash': 0.01, 'spec': 0.35, 'gold': 0.20, 'reits': 0.15}

alloc_current = {'stocks': 0.53, 'spec': 0.23, 'gold': 0.01, 'bonds': 0.03, 'cash': 0.19, 'reits': 0.01}
alloc_std = {'stocks': 0.60, 'bonds': 0.20, 'cash': 0.20, 'spec': 0.0, 'gold': 0.0, 'reits': 0.0}

root = tk.Tk()
root.title("Portfolio Monte Carlo Simulation")

tabControl = ttk.Notebook(root)

# Main tab for core inputs
tab_main = ttk.Frame(tabControl)
tabControl.add(tab_main, text='Main Inputs')

# Debt tab for mortgage, student loan, credit card
tab_debt = ttk.Frame(tabControl)
tabControl.add(tab_debt, text='Debt Inputs')

# Allocations tab for asset allocations and returns
tab_alloc = ttk.Frame(tabControl)
tabControl.add(tab_alloc, text='Allocations')

tabControl.pack(expand=1, fill="both")

inputs = {}
debt_inputs = {}

def add_input(parent, row, label, default_value):
    tk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
    entry = tk.Entry(parent)
    entry.insert(0, str(default_value))
    entry.grid(row=row, column=1, padx=5, pady=2)
    return entry

# --- Main Inputs tab ---

inputs["Starting Age"] = add_input(tab_main, 0, "Starting Age", 36)
inputs["Ending Age"] = add_input(tab_main, 1, "Ending Age", 100)
inputs["Cutoff Age"] = add_input(tab_main, 2, "Cutoff Age", 67)
inputs["Initial Investable Wealth"] = add_input(tab_main, 3, "Initial Investable Wealth", 300000)
inputs["Annual Spending"] = add_input(tab_main, 4, "Annual Spending", 90000)
inputs["Reduced Spending"] = add_input(tab_main, 5, "Reduced Spending", 76000)
inputs["Reduced Spending Start Age"] = add_input(tab_main, 6, "Reduced Spending Start Age", 56)
inputs["Inflation Rate"] = add_input(tab_main, 7, "Inflation Rate", 0.025)
inputs["Social Security Income"] = add_input(tab_main, 8, "Social Security Income", 20000)
inputs["Social Security Start Age"] = add_input(tab_main, 9, "Social Security Start Age", 67)
inputs["Stagflation Years (comma-separated)"] = add_input(tab_main, 10, "Stagflation Years (comma-separated)", "2026")
inputs["Recession Years (comma-separated)"] = add_input(tab_main, 11, "Recession Years (comma-separated)", "2036,2050,2065")
inputs["Employed Income"] = add_input(tab_main, 12, "Employed Income", 120000)
inputs["Employed Income Start Age"] = add_input(tab_main, 13, "Employed Income Start Age", 36)
inputs["Employed Income End Age"] = add_input(tab_main, 14, "Employed Income End Age", 67)
inputs["Non-Employed Income"] = add_input(tab_main, 15, "Non-Employed Income", 10000)
inputs["Non-Employed Income Start Age"] = add_input(tab_main, 16, "Non-Employed Income Start Age", 68)
inputs["Non-Employed Income End Age"] = add_input(tab_main, 17, "Non-Employed Income End Age", 100)
inputs["Property Income"] = add_input(tab_main, 18, "Property Income", 25000)
inputs["Property Income Start Age"] = add_input(tab_main, 19, "Property Income Start Age", 50)
inputs["Start Calendar Year"] = add_input(tab_main, 20, "Start Calendar Year", 2025)
# Note below income inputs
tk.Label(tab_main, text="*incomes are after taxes", fg="gray", font=("Arial", 9, "italic")).grid(
    row=21, column=0, columnspan=2, sticky='w', padx=5, pady=(2, 10)
)

# --- Allocations tab ---

asset_classes = ['stocks', 'spec', 'gold', 'bonds', 'cash', 'reits']
allocation_entries = {}
return_entries = {}

# Header labels
tk.Label(tab_alloc, text="Asset", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10, pady=(10, 5))
tk.Label(tab_alloc, text="Allocation", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=10, pady=(10, 5))
tk.Label(tab_alloc, text="Growth Rate", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=10, pady=(10, 5))

# Data rows
for i, asset in enumerate(asset_classes):
    tk.Label(tab_alloc, text=asset.capitalize()).grid(row=i+1, column=0, sticky='w', padx=10, pady=2)

    alloc_entry = tk.Entry(tab_alloc, width=10)
    alloc_entry.insert(0, str(round(alloc_current.get(asset, 0), 2)))
    alloc_entry.grid(row=i+1, column=1, pady=2)
    allocation_entries[asset] = alloc_entry

    ret_entry = tk.Entry(tab_alloc, width=10)
    ret_entry.insert(0, str(round(normal_returns.get(asset, 0), 3)))
    ret_entry.grid(row=i+1, column=2, pady=2)
    return_entries[asset] = ret_entry

# Add Account Allocation section
account_allocation_labels = ['Taxable', 'Traditional', 'Roth']
account_allocation_keys = ['taxable', 'traditional', 'roth']
account_allocation_entries = {}

# Leave one row empty for spacing
start_row = len(asset_classes) + 2

tk.Label(tab_alloc, text="Account Allocation", font=('Arial', 10, 'bold')).grid(row=start_row, column=0, columnspan=2, pady=(20, 5), sticky="w")

for i, label in enumerate(account_allocation_labels):
    tk.Label(tab_alloc, text=f"{label} (%)").grid(row=start_row + i + 1, column=0, sticky="e", padx=10, pady=2)
    entry = tk.Entry(tab_alloc, width=10)
    entry.insert(0, {"Taxable": "30", "Traditional": "50", "Roth": "20"}[label])  # Default values
    entry.grid(row=start_row + i + 1, column=1, pady=2)
    account_allocation_entries[account_allocation_keys[i]] = entry

# --- Debt Inputs tab ---

debt_labels = [
    ("Mortgage Principal", 0, 700000),
    ("Mortgage Interest Rate", 1, 0.05),
    ("Mortgage Start Age", 2, 30),
    ("Mortgage Paid Off Age", 3, 60),
    ("Student Debt Principal", 4, 0),
    ("Student Debt Interest Rate", 5, 0.065),
    ("Student Debt Start Age", 6, 22),
    ("Student Debt Paid Off Age", 7, 30),
    ("Credit Card Debt Principal", 8, 0),
    ("Credit Card Debt Interest Rate", 9, 0.18),
    ("Credit Card Debt Start Age", 10, 40),
    ("Credit Card Debt Paid Off Age", 11, 46)
]

for label, row, default in debt_labels:
    debt_inputs[label] = add_input(tab_debt, row, label, default)

# --- Functions ---

def generate_spending_curve():
    start_age = int(inputs["Starting Age"].get())
    end_age = int(inputs["Ending Age"].get())
    mortgage_off = int(debt_inputs["Mortgage Paid Off Age"].get())
    inflation = float(inputs["Inflation Rate"].get())
    annual_spend = float(inputs["Annual Spending"].get())
    reduced_spend = float(inputs["Reduced Spending"].get())

    ages = np.arange(start_age, end_age + 1)
    reduced_spending_age = int(inputs["Reduced Spending Start Age"].get())
   
    spending = np.array([
    annual_spend * ((1 + inflation) ** (age - start_age)) if age < reduced_spending_age
    else reduced_spend * ((1 + inflation) ** (age - start_age))
    for age in ages
])

    # Subtract debt payments from spending
    debt_payments = np.zeros_like(ages, dtype=float)

    # Mortgage payments (simple principal + interest, annual)
    mort_principal = float(debt_inputs["Mortgage Principal"].get())
    mort_rate = float(debt_inputs["Mortgage Interest Rate"].get())
    mort_start = int(debt_inputs["Mortgage Start Age"].get())
    mort_off = int(debt_inputs["Mortgage Paid Off Age"].get())
    mort_years = mort_off - mort_start if mort_off > mort_start else 1
    if mort_years > 0:
        mort_payment = mort_principal * (mort_rate * (1 + mort_rate)**mort_years) / ((1 + mort_rate)**mort_years - 1)
    else:
        mort_payment = 0
    for i, age in enumerate(ages):
        if mort_start <= age < mort_off:
            debt_payments[i] += mort_payment

    # Student debt payments (simple amortized)
    stud_principal = float(debt_inputs["Student Debt Principal"].get())
    stud_rate = float(debt_inputs["Student Debt Interest Rate"].get())
    stud_start = int(debt_inputs["Student Debt Start Age"].get())
    stud_off = int(debt_inputs["Student Debt Paid Off Age"].get())
    stud_years = stud_off - stud_start if stud_off > stud_start else 1
    if stud_years > 0:
        stud_payment = stud_principal * (stud_rate * (1 + stud_rate)**stud_years) / ((1 + stud_rate)**stud_years - 1)
    else:
        stud_payment = 0
    for i, age in enumerate(ages):
        if stud_start <= age < stud_off:
            debt_payments[i] += stud_payment

    # Credit card debt payments (simple amortized)
    cc_principal = float(debt_inputs["Credit Card Debt Principal"].get())
    cc_rate = float(debt_inputs["Credit Card Debt Interest Rate"].get())
    cc_start = int(debt_inputs["Credit Card Debt Start Age"].get())
    cc_off = int(debt_inputs["Credit Card Debt Paid Off Age"].get())
    cc_years = cc_off - cc_start if cc_off > cc_start else 1
    if cc_years > 0:
        cc_payment = cc_principal * (cc_rate * (1 + cc_rate)**cc_years) / ((1 + cc_rate)**cc_years - 1)
    else:
        cc_payment = 0
    for i, age in enumerate(ages):
        if cc_start <= age < cc_off:
           debt_payments[i] += cc_payment

    spending += debt_payments
    spending[spending < 0] = 0 # Spending can't be negative

    return ages, spending

def generate_income_curve():
    start_age = int(inputs["Starting Age"].get())
    end_age = int(inputs["Ending Age"].get())
    inflation = float(inputs["Inflation Rate"].get())

    employed_income = float(inputs["Employed Income"].get())
    employed_start = int(inputs["Employed Income Start Age"].get())
    employed_end = int(inputs["Employed Income End Age"].get())

    nonemployed_income = float(inputs["Non-Employed Income"].get())
    nonemployed_start = int(inputs["Non-Employed Income Start Age"].get())
    nonemployed_end = int(inputs["Non-Employed Income End Age"].get())

    social_security_income = float(inputs["Social Security Income"].get())
    ss_start = int(inputs["Social Security Start Age"].get())

    property_income = float(inputs["Property Income"].get())
    prop_start = int(inputs["Property Income Start Age"].get())

    ages = np.arange(start_age, end_age + 1)
    income = np.zeros_like(ages, dtype=float)

    for i, age in enumerate(ages):
        inc = 0
        if employed_start <= age <= employed_end:
             inc += employed_income * ((1 + inflation) ** (age - start_age))
        if nonemployed_start <= age <= nonemployed_end:
           inc += nonemployed_income * ((1 + inflation) ** (age - start_age))
        if age >= ss_start:
            inc += social_security_income * ((1 + inflation) ** (age - start_age))
        if age >= prop_start:
            inc += property_income * ((1 + inflation) ** (age - start_age))
        income[i] = inc

    return ages, income

def parse_years(year_string):
    try:
        years = [int(y.strip()) for y in year_string.split(',') if y.strip()]
        return years
    except Exception:
        return []

# ... (rest of the code remains unchanged above simulate_portfolio)

def simulate_portfolio():
    start_age = int(inputs["Starting Age"].get())
    end_age = int(inputs["Ending Age"].get())
    cutoff_age = int(inputs["Cutoff Age"].get())
    inflation = float(inputs["Inflation Rate"].get())
    initial_wealth = float(inputs["Initial Investable Wealth"].get())
    start_calendar_year = int(inputs["Start Calendar Year"].get())

    stagflation_years = parse_years(inputs["Stagflation Years (comma-separated)"].get())
    recession_years = parse_years(inputs["Recession Years (comma-separated)"].get())

    min_return_floor = {
        'stocks': -0.60,
        'spec': -1.00,
        'bonds': -0.20,
        'cash': -0.05,
        'gold': -0.50,
        'reits': -0.80
    }
    account_allocation = {
    k: float(account_allocation_entries[k].get()) / 100
    for k in ['taxable', 'traditional', 'roth']
    }

    if not np.isclose(sum(account_allocation.values()), 1.0):
        messagebox.showerror("Error", "Account allocation must sum to 100%")
        return

    tax_rates = {
        'capital_gains': 0.15,
        'income': 0.22
    }

    portfolio_by_account = {
        'taxable': initial_wealth * account_allocation['taxable'],
        'traditional': initial_wealth * account_allocation['traditional'],
        'roth': initial_wealth * account_allocation['roth']
    }

    allocs = {}
    rets = {}
    for asset in asset_classes:
        try:
            allocs[asset] = float(allocation_entries[asset].get())
            rets[asset] = float(return_entries[asset].get())
        except Exception:
            messagebox.showerror("Error", f"Invalid allocation or return for {asset}")
            return

    total_alloc = sum(allocs.values())
    if total_alloc == 0:
        messagebox.showerror("Error", "Allocations sum to zero.")
        return
    for k in allocs:
        allocs[k] /= total_alloc

    ages = np.arange(start_age, end_age + 1)
    _, spending_curve = generate_spending_curve()
    _, income_curve = generate_income_curve()

    sim_years = end_age - start_age + 1
    sim_results = np.zeros((num_simulations, sim_years))
    sim_results_real = np.zeros((num_simulations, sim_years))
    inflation_factors = np.array([(1 + inflation) ** i for i in range(sim_years)])

    for sim in range(num_simulations):
        # Clone initial portfolio for each simulation
        portfolio = portfolio_by_account.copy()
        yearly_values = []

        for year_idx, age in enumerate(ages):
            current_year = start_calendar_year + (age - start_age)

            rets_sim = {}
            for asset in asset_classes:
                base_ret = rets[asset]

                if current_year in recession_years:
                    if asset == 'stocks':
                        ret = base_ret - 0.30
                    elif asset == 'spec':
                        ret = base_ret - 0.40
                    else:
                        ret = base_ret
                elif current_year in stagflation_years:
                    if asset in ['stocks', 'bonds']:
                        ret = base_ret - 0.05
                    else:
                        ret = base_ret
                else:
                    ret = base_ret

                ret += np.random.standard_t(fat_tail_df) * normal_volatility[asset]
                ret = max(ret, min_return_floor.get(asset, -1.0))
                rets_sim[asset] = ret

            # Apply returns by account
            for acc in portfolio:
                account_return = 0
                for asset in asset_classes:
                    r = rets_sim[asset]
                    alloc = allocs[asset]
                    if acc == 'taxable':
                        r *= (1 - tax_rates['capital_gains'])  # tax drag
                    account_return += alloc * r
                portfolio[acc] *= (1 + account_return)
                portfolio[acc] = max(0, portfolio[acc])

            # Handle net cashflow (income - spending)
            net_cashflow = income_curve[year_idx] - spending_curve[year_idx]

            if net_cashflow >= 0:
                # Add positive cashflow proportionally
                total_value = sum(portfolio.values())
                for acc in portfolio:
                    portfolio[acc] += net_cashflow * (portfolio[acc] / total_value) if total_value > 0 else 0
            else:
                withdrawal = -net_cashflow
                for acc in ['roth', 'taxable', 'traditional']:
                    if withdrawal <= 0:
                        break
                    if portfolio[acc] <= 0:
                        continue

                    if acc == 'traditional':
                        gross = withdrawal / (1 - tax_rates['income'])
                        taken = min(gross, portfolio[acc])
                        portfolio[acc] -= taken
                        withdrawal -= taken * (1 - tax_rates['income'])

                    elif acc == 'taxable':
                        gross = withdrawal / (1 - tax_rates['capital_gains'])
                        taken = min(gross, portfolio[acc])
                        portfolio[acc] -= taken
                        withdrawal -= taken * (1 - tax_rates['capital_gains'])

                    else:  # roth
                        taken = min(withdrawal, portfolio[acc])
                        portfolio[acc] -= taken
                        withdrawal -= taken


            # Compute portfolio return
            portfolio_value = sum(portfolio.values())
            yearly_values.append(portfolio_value)


        sim_results[sim, :] = yearly_values
        sim_results_real[sim, :] = yearly_values / inflation_factors
    ending_real_wealth = sim_results_real[:, -1]
    cutoff_idx = cutoff_age - start_age
    median_balance_at_retirement = np.percentile(sim_results_real[:, cutoff_idx], 50)
    probability_of_depletion = np.mean(ending_real_wealth <= 0)

    print(f"Probability of portfolio depletion before death: {probability_of_depletion*100:.2f}%")
    print(f"Median real portfolio value at retirement age ({cutoff_age}): ${median_balance_at_retirement:,.2f}")

    pct_25 = np.percentile(sim_results, 25, axis=0)
    pct_50 = np.percentile(sim_results, 50, axis=0)
    pct_75 = np.percentile(sim_results, 75, axis=0)

    pct_25_real = np.percentile(sim_results_real, 25, axis=0)
    pct_50_real = np.percentile(sim_results_real, 50, axis=0)
    pct_75_real = np.percentile(sim_results_real, 75, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()
    

# Existing two plots:
    axs[0].plot(ages[:cutoff_idx+1], pct_25[:cutoff_idx+1], label="25th percentile nominal", color='red', linestyle='--')
    axs[0].plot(ages[:cutoff_idx+1], pct_50[:cutoff_idx+1], label="50th percentile nominal", color='black')
    axs[0].plot(ages[:cutoff_idx+1], pct_75[:cutoff_idx+1], label="75th percentile nominal", color='green', linestyle='--')
    axs[0].plot(ages[:cutoff_idx+1], pct_25_real[:cutoff_idx+1], label="25th percentile real", color='red', linestyle=':')
    axs[0].plot(ages[:cutoff_idx+1], pct_50_real[:cutoff_idx+1], label="50th percentile real", color='black', linestyle=':')
    axs[0].plot(ages[:cutoff_idx+1], pct_75_real[:cutoff_idx+1], label="75th percentile real", color='green', linestyle=':')
    axs[0].set_title(f'Portfolio Value up to Age {cutoff_age}')
    axs[0].set_xlabel('Age')
    axs[0].set_ylabel('Portfolio Value ($)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(ages, pct_25, label="25th percentile nominal", color='red', linestyle='--')
    axs[1].plot(ages, pct_50, label="50th percentile nominal", color='black')
    axs[1].plot(ages, pct_75, label="75th percentile nominal", color='green', linestyle='--')
    axs[1].plot(ages, pct_25_real, label="25th percentile real", color='red', linestyle=':')
    axs[1].plot(ages, pct_50_real, label="50th percentile real", color='black', linestyle=':')
    axs[1].plot(ages, pct_75_real, label="75th percentile real", color='green', linestyle=':')
    axs[1].set_title('Portfolio Value up to Age 100')
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Portfolio Value ($)')
    axs[1].legend()
    axs[1].grid(True)

# New third plot: Real values only, full range
    axs[2].plot(ages, pct_25_real, label="25th percentile real", color='red', linestyle='-')
    axs[2].plot(ages, pct_50_real, label="50th percentile real", color='black', linestyle='-')
    axs[2].plot(ages, pct_75_real, label="75th percentile real", color='green', linestyle='-')
    axs[2].set_title('Real Portfolio Value (Inflation-Adjusted)')
    axs[2].set_xlabel('Age')
    axs[2].set_ylabel('Portfolio Value ($, Real)')
    axs[2].legend()
    axs[2].grid(True)

# 4. Histogram of final wealth
    axs[3].hist(ending_real_wealth, bins=50, color='skyblue', edgecolor='black')
    axs[3].axvline(np.percentile(ending_real_wealth, 50), color='red', linestyle='dashed', label='Median')
    axs[3].set_title(f'Histogram of Real Portfolio Value at Age {end_age}')
    axs[3].set_xlabel('Final Real Portfolio Value ($)')
    axs[3].set_ylabel('Frequency')
    prob_text = f"Depletion Risk: {probability_of_depletion * 100:.2f}%"
    axs[3].text(0.95, 0.95, prob_text, transform=axs[3].transAxes,
            fontsize=10, color='darkred', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    axs[3].legend()
    axs[3].grid(True)



    plt.tight_layout()
    plt.show()

    results = {
        "ages": ages.tolist(),
        "pct_10_nominal": pct_25.tolist(),
        "pct_50_nominal": pct_50.tolist(),
        "pct_90_nominal": pct_75.tolist(),
        "pct_10_real": pct_25_real.tolist(),
        "pct_50_real": pct_50_real.tolist(),
        "pct_90_real": pct_75_real.tolist(),
        "sim_results_nominal": sim_results.tolist(),
        "sim_results_real": sim_results_real.tolist(),
        "probability_of_depletion": probability_of_depletion,
        "median_balance_at_retirement": median_balance_at_retirement,
        "ending_real_wealth": ending_real_wealth.tolist()
    }

    
    return results


def save_results_to_csv(results):
    # Save percentiles nominal and real in CSV
    save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Age", "25th Percentile Nominal", "50th Percentile Nominal", "75th Percentile Nominal",
                  "25th Percentile Real", "50th Percentile Real", "75th Percentile Real"]
        writer.writerow(header)
        for i in range(len(results["ages"])):
            writer.writerow([
                results["ages"][i],
                results["pct_25_nominal"][i],
                results["pct_50_nominal"][i],
                results["pct_75_nominal"][i],
                results["pct_25_real"][i],
                results["pct_50_real"][i],
                results["pct_75_real"][i],
        ])
        messagebox.showinfo("Saved", f"Results saved to {save_path}")

def save_inputs_to_csv():
    save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Key", "Value"])

        for k, e in inputs.items():
            writer.writerow(["inputs", k, e.get()])
        for k, e in debt_inputs.items():
            writer.writerow(["debt_inputs", k, e.get()])
        for k, e in allocation_entries.items():
            writer.writerow(["allocations", k, e.get()])
        for k, e in return_entries.items():
            writer.writerow(["returns", k, e.get()])
        for k, e in account_allocation_entries.items():
            writer.writerow(["account_allocation", k, e.get()])

    messagebox.showinfo("Saved", f"Inputs saved to {save_path}")


    
def load_inputs_from_csv():
    load_path = filedialog.askopenfilename(defaultextension=".csv",
                                           filetypes=[("CSV files", "*.csv")])
    if not load_path:
        return

    data = {
        "inputs": {},
        "debt_inputs": {},
        "allocations": {},
        "returns": {},
        "account_allocation": {}
    }

    with open(load_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for section, key, value in reader:
            data[section][key] = value

    for k, e in inputs.items():
        if k in data["inputs"]:
            e.delete(0, tk.END)
            e.insert(0, data["inputs"][k])
    for k, e in debt_inputs.items():
        if k in data["debt_inputs"]:
            e.delete(0, tk.END)
            e.insert(0, data["debt_inputs"][k])
    for k, e in allocation_entries.items():
        if k in data["allocations"]:
            e.delete(0, tk.END)
            e.insert(0, data["allocations"][k])
    for k, e in return_entries.items():
        if k in data["returns"]:
            e.delete(0, tk.END)
            e.insert(0, data["returns"][k])
    for k, e in account_allocation_entries.items():
        if k in data["account_allocation"]:
            e.delete(0, tk.END)
            e.insert(0, data["account_allocation"][k])

    messagebox.showinfo("Loaded", f"Inputs loaded from {load_path}")


def run_simulation():
    try:
        results = simulate_portfolio()
        root.results = results # store results globally for save
    except Exception as e:
        messagebox.showerror("Error", str(e))
    
def save_results_wrapper():
    if hasattr(root, 'results'):
        save_results_to_csv(root.results)
    else:
        messagebox.showerror("Error", "No simulation results to save. Please run simulation first.")

# --- Buttons ---

btn_run = tk.Button(root, text="Run Simulation", command=run_simulation)
btn_run.pack(pady=5)

btn_save_inputs = tk.Button(root, text="Save Inputs to CSV", command=save_inputs_to_csv)
btn_save_inputs.pack(pady=5)

btn_load_inputs = tk.Button(root, text="Load Inputs from CSV", command=load_inputs_from_csv)
btn_load_inputs.pack(pady=5)

btn_save_results = tk.Button(root, text="Save Results to CSV", command=save_results_wrapper)
btn_save_results.pack(pady=5)
root.mainloop()