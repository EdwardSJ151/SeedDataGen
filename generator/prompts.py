# Prompts used by GENERATOR phases.

# STEP 1 — QA generation prompt (system-only; model outputs QA pairs)
QA_GENERATION_PROMPT = """\
Tarefa:
Leia o texto, faça perguntas e responda a elas.

Siga estas instruções:
1. Faça perguntas diversas que cubram diferentes aspectos do texto.
2. Use diferentes tipos de pergunta:
  - sim/não
  - abertas (o que, como, por que, quando, onde)
  - comparação
  - compreensão
3. Foque em informações factuais e importantes.
4. Perguntas: claras e concisas.
5. Respostas: claras e completas para serem úteis como resposta de um assistente, sem ser prolixo.
6. Use texto simples e linguagem acessível.
7. Cada par deve estar em uma linha:
   "Pergunta: ... Resposta: ..."

Texto:
{sample_text}

Tarefa:
Gere até 5 pares pergunta-resposta."""


# QA_GEN_VAR — Style-constrained QA generation (one pair per style per document)
QA_GEN_VAR_SYSTEM_PROMPT = """\
Leia o texto abaixo e gere exatamente um par pergunta-resposta.

Instruções de estilo:
{style_instruction}

Regras gerais:
1. Foque em informações factuais e importantes do texto.
2. A resposta deve ser clara, completa e útil como resposta de um assistente.
3. Use texto simples e linguagem acessível.
4. Responda em português.
5. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado."""


# QA_LOCAL_MULTIHOP — placeholder prompt for adjacent-chunk QA generation.
# Context spans multiple adjacent passages joined in {sample_text}.
QA_LOCAL_MULTIHOP_PROMPT = """\
[PLACEHOLDER] QA_LOCAL_MULTIHOP_PROMPT

Instruções de estilo:
{style_instruction}

Texto (múltiplos trechos adjacentes):
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado.
Formate assim:
Pergunta: ...
Resposta: ..."""


# QA_SIMILARITY_MULTIHOP — placeholder prompt for similarity-grouped chunks.
# POSITIVE: high-similarity (above) chunk groups.
QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE = """\
[PLACEHOLDER] QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE

Instruções de estilo:
{style_instruction}

Texto (trechos relacionados):
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado.
Formate assim:
Pergunta: ...
Resposta: ..."""


# NEGATIVE: low-similarity (below / range) chunk groups for negative samples.
QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE = """\
[PLACEHOLDER] QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE

Instruções de estilo:
{style_instruction}

Texto (trechos pouco relacionados):
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado.
Formate assim:
Pergunta: ...
Resposta: ..."""


# Maps style name → Portuguese instruction string injected into QA prompts.
QA_GEN_VAR_STYLE_INSTRUCTIONS: dict[str, str] = {
    "general": (
        "Faça uma pergunta geral que aborde o tema principal do texto. "
        "A pergunta deve ser respondível por qualquer pessoa que leu o texto inteiro."
    ),
    "specific": (
        "Faça uma pergunta focada em um detalhe específico, fato pontual ou dado concreto "
        "presente no texto. Evite perguntas amplas."
    ),
    "compositional": (
        "Faça uma pergunta que exija combinar ou relacionar duas ou mais informações "
        "distintas presentes no texto para ser respondida corretamente."
    ),
    "comparative": (
        "Faça uma pergunta que compare dois conceitos, entidades, períodos ou processos "
        "mencionados no texto, destacando semelhanças ou diferenças."
    ),
    "easy": (
        "Nível fácil: a pergunta deve ser direta e a resposta deve exigir pouca elaboração "
        "— em geral, localizar ou reformular uma ou duas ideias explícitas no texto, sem "
        "encadeamentos longos. Tudo deve estar claramente sustentado pelo texto."
    ),
    "medium": (
        "Nível médio: a pergunta pode exigir organizar informações espalhadas no texto ou "
        "um passo simples de inferência, mas sem argumentação longa. A resposta ainda deve "
        "ser construída só com o que o texto permite."
    ),
    "hard": (
        "Nível difícil: a pergunta deve exigir raciocínio mais denso — por exemplo, "
        "relacionar várias partes do texto, explicar um \"porquê\" ou um mecanismo implícito, "
        "ou sintetizar implicações sempre com base no que está escrito, sem extrapolar "
        "além do texto."
    ),
    "extra_hard": (
        "Nível extra difícil: a pergunta deve ser a mais exigente possível em termos de "
        "complexidade da resposta (múltiplos passos, sutilezas, comparações finas ou "
        "estruturação cuidadosa), desde que a resposta completa ainda possa ser fundamentada "
        "no texto. Não invente lacunas: se o texto não suporta um grau tão alto, aproxime-se "
        "do máximo que o texto ainda permita com rigor."
    ),
}
