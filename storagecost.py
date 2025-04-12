import matplotlib.pyplot as plt
import numpy as np

def bitcoin_storage_cost(data_size_kb, btc_price_usd, fee_rate_sat_vb=1):
    vbytes_per_kb = 1024  # 1 KB = 1024 vBytes
    sats_per_btc = 100_000_000  # 1 BTC = 100,000,000 satoshis

    total_vbytes = data_size_kb * vbytes_per_kb
    total_sats = total_vbytes * fee_rate_sat_vb
    cost_btc = total_sats / sats_per_btc
    cost_usd = cost_btc * btc_price_usd

    return cost_usd

def s3_storage_cost(data_size_kb):
    data_size_gb = data_size_kb / (1024 * 1024)  # Convert KB to GB
    s3_cost_per_gb_month = 0.023  # AWS S3 Standard Storage cost per GB per month

    return data_size_gb * s3_cost_per_gb_month

def main():
    data_sizes_kb = np.logspace(0, 6, 7)  # 1KB to 1GB
    btc_prices = [20000, 40000, 60000, 80000, 100000]

    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(8, 5))

    for btc_price in btc_prices:
        btc_costs = [bitcoin_storage_cost(size, btc_price) for size in data_sizes_kb]
        plt.loglog(data_sizes_kb, btc_costs, marker='o', label=f'Bitcoin @ ${btc_price:,}')

    s3_costs = [s3_storage_cost(size) for size in data_sizes_kb]
    plt.loglog(data_sizes_kb, s3_costs, marker='s', linestyle='--', color='black', label='AWS S3')

    plt.xlabel('Data Size (KB)')
    plt.ylabel('Cost (USD)')
    plt.title('Storage cost: bitcoin vs AWS S3')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Format y-axis with dollar signs
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}' if x < 10 else f'${int(x)}' if x < 1000 else f'${x/1000:.1f}k'))

    data_size_labels = ['1KB', '10KB', '100KB', '1MB', '10MB', '100MB', '1GB']
    plt.xticks(data_sizes_kb, data_size_labels)

    break_even_points = []
    for btc_price in btc_prices:
        for i in range(len(data_sizes_kb) - 1):
            btc_cost1 = bitcoin_storage_cost(data_sizes_kb[i], btc_price)
            btc_cost2 = bitcoin_storage_cost(data_sizes_kb[i+1], btc_price)
            s3_cost1 = s3_storage_cost(data_sizes_kb[i])
            s3_cost2 = s3_storage_cost(data_sizes_kb[i+1])

            if (btc_cost1 <= s3_cost1 and btc_cost2 >= s3_cost2) or (btc_cost1 >= s3_cost1 and btc_cost2 <= s3_cost2):
                break_even_points.append((btc_price, data_sizes_kb[i], data_sizes_kb[i+1]))

    for point in break_even_points:
        print(f"At BTC price ${point[0]:,}, break-even point between {point[1]:.2f}KB and {point[2]:.2f}KB")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
