# Consulta a PDF com LangChain e pgVector

Aplicação Python que extrai texto de um PDF, armazena seus embeddings no PostgreSQL e responde perguntas no terminal usando os trechos recuperados.

Base: [devfullcycle/mba-ia-desafio-ingestao-busca](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca), commit `dba3926db2ef4b985569c51d5c8ace6e2c571b06`. O `document.pdf` e o `docker-compose.yml` são os arquivos originais do exemplo.

## Pré-requisitos

- Python 3.12 (versão usada na validação).
- Docker e Docker Compose v2 em execução; porta 5432 disponível.
- `OPENAI_API_KEY` no ambiente, com acesso aos modelos indicados e saldo na API.

## Instalação

Na pasta do projeto, crie e ative o ambiente virtual **antes** de instalar:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Se `python3` não for Python 3.12, use `python3.12 -m venv venv`. No Windows, a ativação é `venv\Scripts\activate`.

A chave já exportada no environment será usada automaticamente; `.env` não sobrescreve variáveis existentes. Se necessário, configure a chave apenas no `.env` local. Esse arquivo e o ambiente virtual são ignorados pelo Git.

| Variável | Padrão | Uso |
|---|---|---|
| `OPENAI_API_KEY` | obrigatório | Autenticação na OpenAI |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Conexão compatível com o Compose |
| `PDF_PATH` | `document.pdf` | Caminho absoluto ou relativo à raiz do projeto |
| `COLLECTION_NAME` | `pdf_documents` | Prefixo da coleção vetorial |

## Execução

Com o ambiente virtual ativo:

```bash
# 1. Banco
 docker compose up -d
 docker compose ps -a
# Aguarde postgres saudável e bootstrap_vector_ext concluído com código 0.

# 2. Ingestão
python src/ingest.py

# 3. Chat
python src/chat.py
```

Digite `sair`, `exit`, `quit`, Ctrl+C ou Ctrl+D para encerrar. Perguntas vazias são ignoradas. Cada pergunta é independente, sem histórico enviado à LLM.

O PDF fornecido contém empresas, faturamentos e anos de fundação. Exemplo baseado nesse PDF:

```text
PERGUNTA: Qual o faturamento da Alfa Agronegócio Indústria?
RESPOSTA: O faturamento da Alfa Agronegócio Indústria é R$ 85.675.568,77.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

A redação da resposta factual pode variar. SuperTechIABrazil e o faturamento de 10 milhões apresentados no enunciado são ilustrativos; a aplicação não insere esses dados no PDF.

## Funcionamento

- `src/ingest.py`: `PyPDFLoader` extrai páginas; `RecursiveCharacterTextSplitter` divide em chunks de até **1000 caracteres**, com **overlap configurado de 150**. Separadores e limites de página podem reduzir a sobreposição efetiva.
- `OpenAIEmbeddings(model="text-embedding-3-small")` gera os vetores; `PGVector` persiste texto, metadados e embeddings.
- A coleção inclui o SHA-256 do PDF. Alterar o PDF seleciona outra coleção; rode a ingestão novamente. Os dados anteriores permanecem armazenados, mas não entram nas consultas ao novo PDF.
- IDs determinísticos evitam duplicação. Uma nova ingestão consulta os IDs existentes e só gera embeddings para os chunks ausentes. Execute uma ingestão por vez.
- `src/search.py` chama `similarity_search_with_score(question, k=10)`. O próprio PGVector vetoriza a pergunta com o mesmo modelo de embeddings. Os scores são distâncias, não probabilidades de a resposta estar correta.
- Os resultados são concatenados no prompt solicitado e enviados a `gpt-5-nano`, com `reasoning_effort="minimal"` e limite de 2048 tokens de conclusão. Uma mensagem de sistema reforça que instruções contidas no PDF não podem alterar as regras.
- Sem contexto, a resposta de ausência de informação é retornada sem chamar a LLM. Respostas vazias ou truncadas também retornam essa mensagem.

## Custos e limites

A primeira ingestão cobra embeddings do texto. Repetições completas não geram novos embeddings. Cada pergunta não vazia gera um embedding e, quando há contexto, uma chamada à LLM. Os trechos recuperados e a pergunta são enviados à OpenAI. Não há chamadas de busca na web.

O prompt restringe o modelo ao contexto, mas uma LLM não oferece garantia absoluta contra alucinações ou instruções maliciosas. A recuperação dos dez trechos também pode deixar uma informação relevante de fora. Valide respostas críticas contra o PDF. PDFs digitalizados sem camada textual precisam de OCR prévio; OCR não está incluído.

## Testes

Testes locais sem custos de API e sem banco:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Validação integrada (usa a API): execute a ingestão duas vezes; a segunda deve informar zero novos chunks. No chat, teste uma empresa do PDF, uma pergunta fora do contexto e uma tentativa de instruir o modelo a ignorar as regras.

## Problemas comuns

- **Conexão recusada:** confira `docker compose ps -a`, aguarde o healthcheck e verifique a porta 5432.
- **Porta ocupada ou container postgres_rag existente:** resolva o conflito ou ajuste o Compose e `DATABASE_URL` para apontarem ao mesmo banco.
- **Erro OpenAI:** verifique chave, acesso aos modelos, saldo, limites e conexão de rede. Erros operacionais são exibidos separadamente da resposta de ausência de informação.
- **Sem resposta após trocar o PDF:** execute `python src/ingest.py` novamente.
- **PDF sem texto:** aplique OCR antes da ingestão.

Para parar o banco preservando os dados: `docker compose down`. O volume persiste entre reinicializações. As credenciais do Compose original são destinadas ao desenvolvimento local.

## Referências

- [Integração PGVector do LangChain](https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector)
- [Modelo GPT-5 nano — OpenAI](https://developers.openai.com/api/docs/models/gpt-5-nano)
