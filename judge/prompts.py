# Prompts used by JUDGE phases.

# STEP 6 — LLM judge prompt (scores a conversation)
JUDGE_PROMPT = """\
Você é um avaliador de qualidade de conversas para treinamento de IA.

Avalie a conversa com base no texto.

Critérios (notas de 1 a 5):
1. Fidelidade ao texto — penalize fortemente qualquer turno que exponha \
metainformação, como "Chunk 19"/"Chunk 73", "com base no texto fornecido", \
"no trecho fornecido", "nos trechos acima" ou "no contexto fornecido"; uma boa \
conversa soa natural e, quando cita a fonte, refere-se ao documento pelo nome.
2. Correção
3. Clareza
4. Coerência — penalize fortemente quando um turno do usuário não for uma \
pergunta (por exemplo, vier formatado como resposta, com título ou itens \
numerados), quebrando os papéis de quem pergunta e de quem responde.
5. Diversidade das interações

Formato (responda exatamente assim, uma nota por linha):

Fidelidade: X
Correção: X
Clareza: X
Coerência: X
Diversidade: X

Texto:
{sample_text}

Conversa:
{conversation}"""
