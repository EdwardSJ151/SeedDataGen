# FULL PIPELINE SPEC

## Overview

For each dataset sample:

```text
sample
→ generate QA pairs (LLM)
→ filter QA (heuristics)
→ for each QA:
      initialize conversation
      expand conversation (LLM loop)
→ filter conversations (heuristics)
→ score conversations (LLM judge)
→ filter by average score > 4.0
→ embedding similarity filter (per sample)
→ output final conversations
```

---

# DATA STRUCTURES

## QA pair

```python
{
    "question": str,
    "answer": str
}
```

## Conversation

```python
{
    "messages": [
        {"role": "user", "content": str},
        {"role": "assistant", "content": str},
        ...
    ]
}
```

---

# STEP 1 — GENERATE QA PAIRS

## Input

```python
sample_text: str
```

## LLM PROMPT

```text
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
4. Use linguagem clara e concisa.
5. Use texto simples.
6. Cada par deve estar em uma linha:
   "Pergunta: ... Resposta: ..."

Texto:
{sample_text}

Tarefa:
Gere até 5 pares pergunta-resposta.
```

## Output parsing

Extract:

```python
qa_pairs = [
    {"question": Q, "answer": A},
    ...
]
```

---

# STEP 2 — FILTER QA PAIRS

## Heuristics

```python
def filter_qa_pairs(qa_pairs):
    filtered = []

    for qa in qa_pairs:
        if len(qa["answer"].strip()) < 10:
            continue

        is_duplicate = False
        for other in filtered:
            if levenshtein(qa["question"], other["question"]) <= 20:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(qa)

    return filtered
```

---

# STEP 3 — INITIALIZE CONVERSATION

```python
def init_conversation(qa):
    return {
        "messages": [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]}
        ]
    }
```

---

# STEP 4 — EXPAND CONVERSATION

## Parameters

```python
N_USER_TURNS = 2 to 4
```

---

## USER TURN PROMPT

```text
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

Gere a próxima mensagem do usuário.

Regras:
1. Continue a conversa atual.
2. Faça uma pergunta, dúvida ou pedido de esclarecimento.
3. Não repita perguntas anteriores.
4. Mantenha-se no conteúdo do texto.
5. Seja natural e breve.
6. Use português.

Texto:
{sample_text}

Histórico:
{conversation_history}

Próxima mensagem do usuário:
```

---

## ASSISTANT TURN PROMPT

```text
Você é um assistente de IA útil e preciso.

Responda à última mensagem do usuário com base no texto abaixo.

Regras:
1. Use apenas informações do texto.
2. Não invente fatos.
3. Seja claro e direto.
4. Responda em português.

Texto:
{sample_text}

Histórico:
{conversation_history}

Resposta:
```

---

## LOOP

```python
def expand_conversation(convo, sample_text):
    for _ in range(N_USER_TURNS):

        user_msg = llm(user_prompt(convo, sample_text))
        convo["messages"].append({"role": "user", "content": user_msg})

        assistant_msg = llm(assistant_prompt(convo, sample_text))
        convo["messages"].append({"role": "assistant", "content": assistant_msg})

    return convo
```

---

# STEP 5 — FILTER CONVERSATIONS (HEURISTICS)

```python
def filter_conversation(convo):

    messages = convo["messages"]

    # 최소 turns
    if len(messages) < 4:
        return False

    # assistant length
    for m in messages:
        if m["role"] == "assistant":
            if len(m["content"].strip()) < 10:
                return False

    # levenshtein user-user
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]

    for i in range(len(user_msgs)):
        for j in range(i+1, len(user_msgs)):
            if levenshtein(user_msgs[i], user_msgs[j]) <= 20:
                return False

    # levenshtein adjacent
    for i in range(len(messages)-1):
        if levenshtein(messages[i]["content"], messages[i+1]["content"]) <= 20:
            return False

    return True
```

---

# STEP 6 — LLM JUDGE

## PROMPT

```text
Você é um avaliador de qualidade de conversas para treinamento de IA.

Avalie a conversa com base no texto.

Critérios:
1. Fidelidade ao texto
2. Correção
3. Clareza
4. Coerência
5. Diversidade das interações

Dê notas de 1 a 5.

Formato:

Fidelidade: X
Correção: X
Clareza: X
Coerência: X
Diversidade: X

Texto:
{sample_text}

Conversa:
{conversation}
```

---

## PARSING

```python
scores = [
    fidelidade,
    correcao,
    clareza,
    coerencia,
    diversidade
]

avg_score = sum(scores) / len(scores)
```

---

## FILTER RULE

```python
if avg_score <= 4.0:
    reject
```

---

# STEP 7 — EMBEDDING FILTER (PER SAMPLE)

## Input

```python
conversations = [C1, C2, C3, ...]
```

---

## Convert to text

```python
def convo_to_text(convo):
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in convo["messages"]
    )
```

---

## Embeddings

```python
embeddings = model.encode([convo_to_text(c) for c in conversations])
```

---

## Similarity filter

```python
THRESHOLD = 0.9

final = []

for i, emb_i in enumerate(embeddings):
    keep = True

    for j, emb_j in enumerate(final):
        if cosine_similarity(emb_i, emb_j) > THRESHOLD:
            keep = False
            break

    if keep:
        final.append(emb_i)
```

(also keep index mapping to conversations)

---

# FINAL OUTPUT

```python
[
    {
        "messages": [...]
    },
    ...
]
```

---

# FULL EXECUTION FLOW

```python
def process_sample(sample_text):

    qa_pairs = generate_qa(sample_text)
    qa_pairs = filter_qa_pairs(qa_pairs)

    conversations = []

    for qa in qa_pairs:
        convo = init_conversation(qa)
        convo = expand_conversation(convo, sample_text)

        if not filter_conversation(convo):
            continue

        score = judge(convo, sample_text)
        if score <= 4.0:
            continue

        conversations.append(convo)

    conversations = embedding_filter(conversations)

    return conversations
```

---
