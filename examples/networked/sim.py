import luckyrobots as lr

if __name__ == '__main__':
    lr.start(
        sim_addr="192.168.1.8",
        sim_port=3000,
        is_sim=True,
        is_policy=False,
    )