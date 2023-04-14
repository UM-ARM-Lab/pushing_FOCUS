from collect_data import collect_data


def main():
    def _nominal_generator(rng):
        vx = rng.uniform(0.02, 0.1)
        while True:
            vy = rng.normal(0, 0.04)
            if abs(vy) > 0.03:
                break
        return vx, vy

    collect_data(_nominal_generator, 'dissimilar', 'target.xml', 100, 0)


if __name__ == '__main__':
    main()
