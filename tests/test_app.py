import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from config import pdf_path
from ingest import ingest, load_chunks
from search import FALLBACK, PDFChat


class AppTests(unittest.TestCase):
    def test_pdf_chunks(self):
        chunks = load_chunks(pdf_path())
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(0 < len(c.page_content) <= 1000 for c in chunks))
        self.assertTrue(all("page" in c.metadata for c in chunks))

    def test_empty_retrieval_does_not_call_llm(self):
        store, llm = Mock(), Mock()
        store.similarity_search_with_score.return_value = []
        self.assertEqual(PDFChat(store, llm).answer("Pergunta?"), FALLBACK)
        llm.invoke.assert_not_called()

    def test_blank_question_does_not_call_services(self):
        store, llm = Mock(), Mock()
        self.assertEqual(PDFChat(store, llm).answer("  "), FALLBACK)
        store.similarity_search_with_score.assert_not_called()
        llm.invoke.assert_not_called()

    def test_context_and_k(self):
        store, llm = Mock(), Mock()
        store.similarity_search_with_score.return_value = [(Document(page_content="Receita: R$ 10."), .1)]
        llm.invoke.return_value = AIMessage(content="R$ 10.")
        self.assertEqual(PDFChat(store, llm).answer("Qual a receita?"), "R$ 10.")
        store.similarity_search_with_score.assert_called_once_with("Qual a receita?", k=10)
        messages = llm.invoke.call_args.args[0]
        self.assertIn("Receita: R$ 10.", messages[1].content)
        self.assertIn(FALLBACK, messages[0].content)

    def test_truncated_answer_is_rejected(self):
        store, llm = Mock(), Mock()
        store.similarity_search_with_score.return_value = [(Document(page_content="Texto"), .1)]
        llm.invoke.return_value = AIMessage(content="Resposta incompleta", response_metadata={"finish_reason": "length"})
        self.assertEqual(PDFChat(store, llm).answer("Pergunta?"), FALLBACK)

    @patch("ingest.vector_store")
    @patch("ingest.load_chunks")
    def test_reingestion_skips_existing_embeddings(self, load, factory):
        load.return_value = [Document(page_content="texto")]
        store = factory.return_value
        store.collection_name = "test"
        store.get_by_ids.return_value = [Document(id="test:1000:150:0", page_content="texto")]
        self.assertEqual(ingest(Path("unused.pdf")), (0, 1))
        store.add_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
