#!/usr/bin/env python
from collect_data import collect_data


def main():
    def _nominal_generator(rng):
        return rng.uniform(0.02, 0.1), rng.normal(0, 0.006)

    collect_data(_nominal_generator, 'similar', 'target.xml', 100, 0)


if __name__ == '__main__':
    main()
