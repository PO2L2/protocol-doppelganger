from __future__ import annotations

import os
import sys


def main() -> int:
    check_only = "--check" in sys.argv
    if check_only:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from digital_twin_game.game import DigitalTwinGame

    game = DigitalTwinGame()
    if check_only:
        game._draw()
        game._present_frame()
        pygame.quit()
        return 0
    game.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
