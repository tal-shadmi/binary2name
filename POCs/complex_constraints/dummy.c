#include <stdio.h>

int checkme(int x) {
    x += 1;
    int y=0;
    if (x < 5) {
        y = 10;
    } else {
        y = 20;
    }
    return y;
}

void main() {
    printf("%d\n", checkme(10));
//}