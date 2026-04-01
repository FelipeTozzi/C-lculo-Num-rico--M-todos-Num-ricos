# Funções auxiliares

def norma_infinito(v):
    return max(abs(x) for x in v)


def erro_relativo(x_novo, x_antigo):
    return norma_infinito([x_novo[i] - x_antigo[i] for i in range(len(x_novo))]) / max(1, norma_infinito(x_novo))



# 1. Eliminação de Gauss

def gauss(A, b):
    n = len(A)

    # Monta matriz aumentada
    M = [A[i] + [b[i]] for i in range(n)]

    # Eliminação
    for k in range(n - 1):
        if M[k][k] == 0:
            raise Exception("Pivô zero encontrado!")

        for i in range(k + 1, n):
            fator = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= fator * M[k][j]

    # Substituição regressiva
    x = [0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - soma) / M[i][i]

    return x


# 2. Decomposição LU

def decomposicao_lu(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):
        # U
        for j in range(i, n):
            soma = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - soma

        # L
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                soma = sum(L[j][k] * U[k][i] for k in range(i))
                if U[i][i] == 0:
                    raise Exception("Divisão por zero!")
                L[j][i] = (A[j][i] - soma) / U[i][i]

    return L, U


def resolver_lu(L, U, b):
    n = len(L)

    # Ly = b (substituição direta)
    y = [0]*n
    for i in range(n):
        soma = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - soma

    # Ux = y (substituição regressiva)
    x = [0]*n
    for i in range(n-1, -1, -1):
        soma = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - soma) / U[i][i]

    return x


# 3. Método de Jacobi

def jacobi(A, b, x0, eps, max_iter=100):
    n = len(A)
    x = x0[:]

    for _ in range(max_iter):
        x_novo = [0]*n

        for i in range(n):
            soma = sum(A[i][j]*x[j] for j in range(n) if j != i)
            x_novo[i] = (b[i] - soma) / A[i][i]

        if erro_relativo(x_novo, x) < eps:
            return x_novo

        x = x_novo

    return x


# 4. Gauss-Seidel

def gauss_seidel(A, b, x0, eps, max_iter=100):
    n = len(A)
    x = x0[:]

    for _ in range(max_iter):
        x_antigo = x[:]

        for i in range(n):
            soma1 = sum(A[i][j]*x[j] for j in range(i))
            soma2 = sum(A[i][j]*x_antigo[j] for j in range(i+1, n))

            x[i] = (b[i] - soma1 - soma2) / A[i][i]

        if erro_relativo(x, x_antigo) < eps:
            return x

    return x


# Programa principal (teste)

if __name__ == "__main__":
    n = int(input("Digite a dimensão n: "))

    print("Digite a matriz A:")
    A = [list(map(float, input().split())) for _ in range(n)]

    print("Digite o vetor b:")
    b = list(map(float, input().split()))

    print("\nEscolha o método:")
    print("1 - Gauss")
    print("2 - LU")
    print("3 - Jacobi")
    print("4 - Gauss-Seidel")

    op = int(input("Opção: "))

    if op == 1:
        x = gauss(A, b)

    elif op == 2:
        L, U = decomposicao_lu(A)
        x = resolver_lu(L, U, b)

    elif op == 3:
        x0 = list(map(float, input("Chute inicial: ").split()))
        eps = float(input("Precisão: "))
        x = jacobi(A, b, x0, eps)

    elif op == 4:
        x0 = list(map(float, input("Chute inicial: ").split()))
        eps = float(input("Precisão: "))
        x = gauss_seidel(A, b, x0, eps)

    else:
        print("Opção inválida!")
        exit()

    print("\nSolução x:")
    for i in range(len(x)):
        print(f"x{i+1} = {x[i]:.6f}")
        