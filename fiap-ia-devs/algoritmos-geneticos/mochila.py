import random

class Produto:
    def __init__(self, nome, espaco, valor):
        self.nome = nome
        self.espaco = espaco
        self.valor = valor

class Individuo:
    def __init__(self, produtos, limite):
        self.produtos = produtos
        self.limite = limite
        self.cromossomo = [random.choice([0, 1]) for _ in range(len(produtos))]
        self.nota_avaliacao = self.avaliacao()
        self.espaco_usado = self.calcular_espaco()

    def avaliacao(self):
        soma_valores = 0
        soma_espacos = 0
        for i in range(len(self.cromossomo)):
            if self.cromossomo[i] == 1:
                soma_valores += self.produtos[i].valor
                soma_espacos += self.produtos[i].espaco
            if soma_espacos > self.limite:
                soma_valores = 1  # Penalização se exceder o limite
        return soma_valores

    def calcular_espaco(self):
        soma_espacos = 0
        for i in range(len(self.cromossomo)):
            if self.cromossomo[i] == 1:
                soma_espacos += self.produtos[i].espaco
        return soma_espacos

class AlgoritmoGenetico:
    def __init__(self, produtos, limite, tamanho_populacao, taxa_mutacao, geracoes):
        self.produtos = produtos
        self.limite = limite
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.geracoes = geracoes
        self.populacao = [Individuo(produtos, limite) for _ in range(tamanho_populacao)]
        self.melhor_solucao = self.populacao[0]  # Inicializa com o primeiro indivíduo

    def ordena_populacao(self):
        self.populacao.sort(key=lambda x: x.nota_avaliacao, reverse=True)

    def seleciona_pai(self):
        soma_avaliacao = sum(ind.nota_avaliacao for ind in self.populacao)
        valor_sorteado = random.uniform(0, soma_avaliacao)
        soma = 0
        for i, ind in enumerate(self.populacao):
            soma += ind.nota_avaliacao
            if soma >= valor_sorteado:
                return i
        return len(self.populacao) - 1

    def crossover(self, pai1, pai2):
        corte = random.randint(1, len(pai1.cromossomo) - 1)
        filho1_cromossomo = pai1.cromossomo[:corte] + pai2.cromossomo[corte:]
        filho2_cromossomo = pai2.cromossomo[:corte] + pai1.cromossomo[corte:]
        return Individuo(self.produtos, self.limite), Individuo(self.produtos, self.limite)

    def mutacao(self, individuo):
        for i in range(len(individuo.cromossomo)):
            if random.random() < self.taxa_mutacao:
                individuo.cromossomo[i] = 1 - individuo.cromossomo[i] # Troca 0 por 1 ou 1 por 0
        individuo.nota_avaliacao = individuo.avaliacao()
        individuo.espaco_usado = individuo.calcular_espaco()

    def visualizar(self, geracao):
        print(f"\nGeração {geracao + 1}:")
        print(f"Melhor Nota: {self.populacao[0].nota_avaliacao:.2f}, Espaço Usado: {self.populacao[0].espaco_usado:.2f}")

    def visualizar_final(self):
        print(f"Melhor solução -> G:{self.geracoes} Valor:{self.melhor_solucao.nota_avaliacao:.2f} Espaço:{self.melhor_solucao.espaco_usado:.2f} Cromossomo:")
        print(self.melhor_solucao.cromossomo)
        for i in range(len(self.produtos)):
            if self.melhor_solucao.cromossomo[i] == 1:
                print(f"Nome: {self.produtos[i].nome} R${self.produtos[i].valor:.2f}")

    def resolver(self):
        for geracao in range(self.geracoes):
            self.ordena_populacao()
            self.visualizar(geracao)  # Imprime a melhor geração atual

            nova_populacao = []
            while len(nova_populacao) < self.tamanho_populacao:
                pai1 = self.seleciona_pai()
                pai2 = self.seleciona_pai()
                filhos = self.crossover(self.populacao[pai1], self.populacao[pai2])
                for filho in filhos:
                    self.mutacao(filho)
                    nova_populacao.append(filho)

            self.populacao = nova_populacao

            # Atualiza a melhor solução se necessário
            if self.populacao[0].nota_avaliacao > self.melhor_solucao.nota_avaliacao:
                self.melhor_solucao = self.populacao[0]

def main():
    produtos = [
        Produto("Geladeira Dako", 1.5, 999.9),
        Produto("Iphone 6", 0.5, 2911.12),
        Produto("TV 50'", 0.8, 3999.9),
        Produto("TV 42'", 0.7, 2999.0),
        Produto("Notebook Dell", 1.2, 2499.9),
        Produto("Microondas Electrolux", 0.6, 308.66),
        Produto("Microondas LG", 0.6, 429.9),
        Produto("Microondas Panasonic", 0.5, 299.29),
        Produto("Notebook Lenovo", 1.1, 1999.9),
        Produto("Notebook Asus", 1.0, 3999.0)
    ]

    limite = 2  # Limite de espaço em metros cúbicos
    tamanho_populacao = 20
    taxa_mutacao = 0.01
    geracoes = 100

    ag = AlgoritmoGenetico(produtos, limite, tamanho_populacao, taxa_mutacao, geracoes)
    ag.resolver()

    print("\nMelhor Solução Final:")
    ag.visualizar_final()  # Imprime a melhor solução final

if __name__ == "__main__":
    main()
