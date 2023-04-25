#!/usr/bin/env python
from collect_data import collect_data


def main():
    def _nominal_generator(rng):
        return rng.uniform(0.02, 0.1), rng.normal(0, 0.04), rng.normal(0, 0.4)

    collect_data(_nominal_generator, 'source', 'source.xml', 1000, 0)


if __name__ == '__main__':
    main()
