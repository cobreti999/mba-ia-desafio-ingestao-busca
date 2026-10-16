"""Chat de terminal: python src/chat.py."""
import sys
from search import PDFChat


def main():
    try:
        chat = PDFChat()
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Não foi possível iniciar. Confira Docker, DATABASE_URL e OPENAI_API_KEY.", file=sys.stderr)
        return 1
    print("Faça sua pergunta (digite sair para encerrar):")
    while True:
        try:
            question = input("PERGUNTA: ").strip()
            if question.casefold() in {"sair", "exit", "quit"}:
                break
            if not question:
                continue
            try:
                print(f"RESPOSTA: {chat.answer(question)}\n\n---\n")
            except Exception:
                print("Erro na consulta. Confira a conexão com o banco e a OpenAI; tente novamente.", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print()
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
