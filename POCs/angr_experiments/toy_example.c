int f(int n) {
    int i = 0;
    while (i < 10) {
        n += i;
        i++;
    }
    if (n > 100) {
        return 1;
    }
    return 0;
}


int main() {
    int result = f(20);
    return result;
}