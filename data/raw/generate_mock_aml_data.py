"""
Synthetic Heterogeneous AML Dataset Generator.

Generates realistic multi-relational financial transaction networks containing:
- High-risk smurfing / structuring topologies (fan-out and fan-in)
- Circular money laundering flows (directed cycles)
- Shared device mule syndicates (emulator/VPN device collusion)
- Realistic licit banking baseline traffic (Pareto amount distribution)

Outputs 6 relational CSV tables:
- accounts.csv
- customers.csv
- devices.csv
- transactions.csv
- customer_accounts.csv
- account_devices.csv
"""

import argparse
import csv
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple


def generate_synthetic_aml_data(
    num_accounts: int = 5000,
    num_customers: int = 4000,
    num_devices: int = 2000,
    num_transactions: int = 30000,
    output_dir: Path = Path("data/raw"),
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate synthetic heterogeneous financial crime dataset.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 1, 1, 0, 0, 0)

    print(f"[*] Initializing AML synthetic generator with seed={seed}...")
    print(f"[*] Target entity counts: {num_accounts} accounts, {num_customers} customers, {num_devices} devices.")

    # ---------------------------------------------------------
    # 1. Generate Customers
    # ---------------------------------------------------------
    customers = []
    customer_ids = [f"CUST_{i:06d}" for i in range(1, num_customers + 1)]
    occupations = ["Engineer", "Accountant", "Teacher", "Merchant", "Consultant", "Student", "Retired", "Unemployed"]

    for cid in customer_ids:
        kyc_level = random.choices([1, 2, 3], weights=[0.15, 0.60, 0.25])[0]
        # Base risk score: mostly low, small tail of high-risk customers
        base_risk = random.betavariate(1.5, 8.0)
        country_risk = 1 if random.random() < 0.05 else 0
        if country_risk == 1:
            base_risk = min(1.0, base_risk + 0.3)

        customers.append({
            "customer_id": cid,
            "kyc_level": kyc_level,
            "risk_score": round(base_risk, 4),
            "country_risk": country_risk,
            "occupation": random.choice(occupations),
        })

    # ---------------------------------------------------------
    # 2. Generate Devices
    # ---------------------------------------------------------
    devices = []
    device_ids = [f"DEV_{i:06d}" for i in range(1, num_devices + 1)]
    device_types = ["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "server_linux"]

    for did in device_ids:
        dtype = random.choices(device_types, weights=[0.42, 0.40, 0.12, 0.05, 0.01])[0]
        is_proxy_or_vpn = 1 if random.random() < 0.08 else 0
        is_emulator = 1 if (dtype.startswith("mobile") and random.random() < 0.03) else 0

        devices.append({
            "device_id": did,
            "device_type": dtype,
            "is_proxy_or_vpn": is_proxy_or_vpn,
            "is_emulator": is_emulator,
        })

    # ---------------------------------------------------------
    # 3. Generate Accounts
    # ---------------------------------------------------------
    account_ids = [f"ACC_{i:06d}" for i in range(1, num_accounts + 1)]
    account_types = ["checking", "savings", "business", "crypto_onramp"]

    # Assign initial structures
    accounts_dict = {}
    for aid in account_ids:
        atype = random.choices(account_types, weights=[0.65, 0.25, 0.08, 0.02])[0]
        init_balance = round(random.expovariate(1 / 5000.0) + 100.0, 2)
        create_delta = timedelta(days=random.randint(30, 1000))
        created_date = (base_time - create_delta).strftime("%Y-%m-%d %H:%M:%S")

        accounts_dict[aid] = {
            "account_id": aid,
            "account_type": atype,
            "initial_balance": init_balance,
            "creation_date": created_date,
            "is_mule": 0,
            "typology": "licit",
        }

    # ---------------------------------------------------------
    # 4. Generate Customer-Account Ownership
    # ---------------------------------------------------------
    customer_accounts = []
    account_to_customer: Dict[str, str] = {}
    cust_idx = 0

    for aid in account_ids:
        # Most customers have 1 account; some have 2; few have 3
        cid = customer_ids[cust_idx % num_customers]
        customer_accounts.append({"customer_id": cid, "account_id": aid})
        account_to_customer[aid] = cid
        if random.random() > 0.35:
            cust_idx += 1

    # ---------------------------------------------------------
    # 5. Generate Account-Device Logins (Default Licit)
    # ---------------------------------------------------------
    account_devices = []
    account_device_map: Dict[str, List[str]] = {aid: [] for aid in account_ids}

    for aid in account_ids:
        # Licit account logs in from 1 or 2 primary devices
        num_devs = 1 if random.random() < 0.8 else 2
        assigned_devs = random.sample(device_ids, num_devs)
        for did in assigned_devs:
            last_login = (base_time - timedelta(hours=random.randint(1, 720))).strftime("%Y-%m-%d %H:%M:%S")
            login_count = random.randint(5, 120)
            account_devices.append({
                "account_id": aid,
                "device_id": did,
                "last_login": last_login,
                "login_count": login_count,
            })
            account_device_map[aid].append(did)

    # ---------------------------------------------------------
    # 6. Inject Financial Crime Typologies (Fraud Rings)
    # ---------------------------------------------------------
    transactions = []
    tx_counter = 1
    mule_accounts: Set[str] = set()

    # Typology A: Smurfing Networks (Fan-Out Structuring followed by Fan-In Aggregation)
    # Typically: Dirty Source -> Smurfs ($8,500 - $9,900 each) -> Aggregator
    num_smurfing_rings = max(3, num_accounts // 500)
    available_accounts = list(account_ids)
    random.shuffle(available_accounts)
    ring_ptr = 0

    print(f"[*] Synthesizing {num_smurfing_rings} Smurfing / Structuring rings...")
    for ring_i in range(num_smurfing_rings):
        if ring_ptr + 12 >= len(available_accounts):
            break
        source_aid = available_accounts[ring_ptr]
        aggregator_aid = available_accounts[ring_ptr + 1]
        smurf_aids = available_accounts[ring_ptr + 2 : ring_ptr + 10]
        ring_ptr += 10

        accounts_dict[source_aid]["is_mule"] = 1
        accounts_dict[source_aid]["typology"] = "smurfing_source"
        accounts_dict[aggregator_aid]["is_mule"] = 1
        accounts_dict[aggregator_aid]["typology"] = "smurfing_aggregator"
        mule_accounts.add(source_aid)
        mule_accounts.add(aggregator_aid)

        # Base timestamp for this smurfing campaign
        ring_time = base_time + timedelta(days=random.randint(1, 60), hours=random.randint(0, 23))

        # Fan-out: Source -> Smurfs (rapid succession within hours)
        for smurf_aid in smurf_aids:
            accounts_dict[smurf_aid]["is_mule"] = 1
            accounts_dict[smurf_aid]["typology"] = "smurfing_mule"
            mule_accounts.add(smurf_aid)

            # Amount structured strictly below $10,000 threshold
            structured_amount = round(random.uniform(8500.0, 9950.0), 2)
            tx_time = ring_time + timedelta(minutes=random.randint(5, 180))

            transactions.append({
                "tx_id": f"TX_{tx_counter:08d}",
                "source_account_id": source_aid,
                "target_account_id": smurf_aid,
                "amount": structured_amount,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": "wire",
                "is_fraudulent": 1,
            })
            tx_counter += 1

            # Fan-in: Smurf -> Aggregator (24-48 hours later, taking ~2% cut)
            forward_amount = round(structured_amount * random.uniform(0.96, 0.98), 2)
            tx_time_forward = tx_time + timedelta(hours=random.randint(18, 48))

            transactions.append({
                "tx_id": f"TX_{tx_counter:08d}",
                "source_account_id": smurf_aid,
                "target_account_id": aggregator_aid,
                "amount": forward_amount,
                "timestamp": tx_time_forward.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": "p2p",
                "is_fraudulent": 1,
            })
            tx_counter += 1

    # Typology B: Circular Laundering Flows (Directed Cycles)
    # e.g., A -> B -> C -> D -> A
    num_cycles = max(3, num_accounts // 600)
    print(f"[*] Synthesizing {num_cycles} Circular Laundering Rings (cycles)...")
    for _ in range(num_cycles):
        if ring_ptr + 5 >= len(available_accounts):
            break
        cycle_len = random.choice([3, 4, 5])
        cycle_aids = available_accounts[ring_ptr : ring_ptr + cycle_len]
        ring_ptr += cycle_len

        for aid in cycle_aids:
            accounts_dict[aid]["is_mule"] = 1
            accounts_dict[aid]["typology"] = "circular_mule"
            mule_accounts.add(aid)

        cycle_time = base_time + timedelta(days=random.randint(10, 80))
        cur_amount = round(random.uniform(15000.0, 45000.0), 2)

        for step in range(cycle_len):
            u = cycle_aids[step]
            v = cycle_aids[(step + 1) % cycle_len]
            step_time = cycle_time + timedelta(hours=step * random.randint(4, 12))
            cur_amount = round(cur_amount * random.uniform(0.97, 0.99), 2)

            transactions.append({
                "tx_id": f"TX_{tx_counter:08d}",
                "source_account_id": u,
                "target_account_id": v,
                "amount": cur_amount,
                "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": "wire",
                "is_fraudulent": 1,
            })
            tx_counter += 1

    # Typology C: Shared Device Mule Syndicate (Botnet / Emulator Collusion)
    # A single proxy/emulator device connects 6-10 separate accounts with different customer IDs
    num_syndicates = max(2, num_accounts // 800)
    print(f"[*] Synthesizing {num_syndicates} Shared Device Fraud Syndicates...")
    proxy_devices = [d for d in devices if d["is_proxy_or_vpn"] == 1 or d["is_emulator"] == 1]
    if not proxy_devices:
        proxy_devices = devices[:5]

    for syn_i in range(num_syndicates):
        if ring_ptr + 7 >= len(available_accounts):
            break
        syndicate_device = proxy_devices[syn_i % len(proxy_devices)]["device_id"]
        syndicate_aids = available_accounts[ring_ptr : ring_ptr + 6]
        ring_ptr += 6

        syndicate_collector = available_accounts[ring_ptr]
        ring_ptr += 1
        accounts_dict[syndicate_collector]["is_mule"] = 1
        accounts_dict[syndicate_collector]["typology"] = "syndicate_cashout"
        mule_accounts.add(syndicate_collector)

        syn_time = base_time + timedelta(days=random.randint(5, 75))

        for aid in syndicate_aids:
            accounts_dict[aid]["is_mule"] = 1
            accounts_dict[aid]["typology"] = "syndicate_mule"
            mule_accounts.add(aid)

            # Bind shared device with high login count
            account_devices.append({
                "account_id": aid,
                "device_id": syndicate_device,
                "last_login": syn_time.strftime("%Y-%m-%d %H:%M:%S"),
                "login_count": random.randint(40, 200),
            })

            # Rapid transfer out to syndicate cashout
            tx_time = syn_time + timedelta(minutes=random.randint(1, 45))
            transactions.append({
                "tx_id": f"TX_{tx_counter:08d}",
                "source_account_id": aid,
                "target_account_id": syndicate_collector,
                "amount": round(random.uniform(4000.0, 12000.0), 2),
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": "swift",
                "is_fraudulent": 1,
            })
            tx_counter += 1

    # ---------------------------------------------------------
    # 7. Generate Baseline Licit Banking Transactions (Pareto / Log-normal)
    # ---------------------------------------------------------
    remaining_tx = max(0, num_transactions - len(transactions))
    print(f"[*] Generating {remaining_tx} licit baseline transactions...")

    # Exclude accounts flagged as purely internal mules from standard licit flows
    licit_accounts = [aid for aid in account_ids if aid not in mule_accounts]
    channels = ["ach", "wire", "p2p", "pos"]

    for _ in range(remaining_tx):
        src = random.choice(licit_accounts)
        dst = random.choice(account_ids)
        if src == dst:
            continue

        # Realistic banking amounts: Log-normal distribution
        # E.g. median ~$65, mean ~$180, rare transactions up to $5,000
        raw_amt = random.lognormvariate(4.0, 1.2)
        amount = round(max(5.0, min(raw_amt, 15000.0)), 2)

        tx_time = base_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        transactions.append({
            "tx_id": f"TX_{tx_counter:08d}",
            "source_account_id": src,
            "target_account_id": dst,
            "amount": amount,
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": random.choices(channels, weights=[0.45, 0.20, 0.30, 0.05])[0],
            "is_fraudulent": 0,
        })
        tx_counter += 1

    # Sort transactions chronologically
    transactions.sort(key=lambda x: x["timestamp"])

    # ---------------------------------------------------------
    # 8. Write to CSV Files
    # ---------------------------------------------------------
    def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"    [+] Saved {len(rows):,} rows -> {path}")

    print("[*] Writing relational CSV tables to disk...")
    write_csv(
        output_dir / "accounts.csv",
        list(accounts_dict.values()),
        ["account_id", "account_type", "initial_balance", "creation_date", "is_mule", "typology"],
    )

    write_csv(
        output_dir / "customers.csv",
        customers,
        ["customer_id", "kyc_level", "risk_score", "country_risk", "occupation"],
    )

    write_csv(
        output_dir / "devices.csv",
        devices,
        ["device_id", "device_type", "is_proxy_or_vpn", "is_emulator"],
    )

    write_csv(
        output_dir / "transactions.csv",
        transactions,
        ["tx_id", "source_account_id", "target_account_id", "amount", "timestamp", "channel", "is_fraudulent"],
    )

    write_csv(
        output_dir / "customer_accounts.csv",
        customer_accounts,
        ["customer_id", "account_id"],
    )

    write_csv(
        output_dir / "account_devices.csv",
        account_devices,
        ["account_id", "device_id", "last_login", "login_count"],
    )

    mule_count = len(mule_accounts)
    fraud_tx_count = sum(1 for tx in transactions if tx["is_fraudulent"] == 1)
    imbalance_rate = (mule_count / num_accounts) * 100.0

    print("\n" + "=" * 60)
    print("GraphSentry Synthetic AML Data Generation Complete:")
    print(f"  Total Accounts    : {num_accounts:,} (Mule: {mule_count:,} | {imbalance_rate:.2f}%)")
    print(f"  Total Customers   : {num_customers:,}")
    print(f"  Total Devices     : {num_devices:,}")
    print(f"  Total Transactions: {len(transactions):,} (Illicit Tx: {fraud_tx_count:,})")
    print("=" * 60 + "\n")

    return {
        "num_accounts": num_accounts,
        "num_mules": mule_count,
        "num_customers": num_customers,
        "num_devices": num_devices,
        "num_transactions": len(transactions),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic multi-relational AML graph dataset.")
    parser.add_argument("--num-accounts", type=int, default=5000, help="Number of accounts to generate")
    parser.add_argument("--num-customers", type=int, default=4000, help="Number of customers")
    parser.add_argument("--num-devices", type=int, default=2000, help="Number of devices")
    parser.add_argument("--num-transactions", type=int, default=30000, help="Total transactions")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    generate_synthetic_aml_data(
        num_accounts=args.num_accounts,
        num_customers=args.num_customers,
        num_devices=args.num_devices,
        num_transactions=args.num_transactions,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
