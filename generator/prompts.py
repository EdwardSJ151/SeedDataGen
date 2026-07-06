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
7. Não use referências meta nas perguntas nem nas respostas (por exemplo "o texto fornecido", "o trecho", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
8. Cada par deve estar em uma linha:
   "Pergunta: ... Resposta: ..."

Texto:
{sample_text}

Tarefa:
Gere até 5 pares pergunta-resposta."""


# QA_GEN_VAR — Style-constrained QA generation (one pair per style per document)
QA_GEN_VAR_SYSTEM_PROMPT = """\
Leia o texto abaixo e gere exatamente um par pergunta-resposta.

Regras gerais:
1. Foque em informações factuais e importantes do texto.
2. A resposta deve ser clara, completa e útil como resposta de um assistente.
3. Use texto simples e linguagem acessível.
4. Não use referências meta na pergunta nem na resposta (por exemplo "o texto fornecido", "o trecho", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
5. Responda em português.
6. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Instruções de estilo:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado, focando no texto e só usando o sumário do texto como referencia/complemento."""


# DOG_INSTRUCT — back-translation style question generation + answer rewrite
DOG_INSTRUCT_QUESTION_SYSTEM_PROMPT = """\
Você é um usuário que faz uma única pergunta a um assistente. Formule a pergunta de \
modo que a resposta natural seja o conteúdo geral de um documento — você quer a ideia \
geral do documento como um todo. Você pode incluir um pequeno complemento ou \
especificar um aspecto na pergunta, desde que a pergunta gerada ainda corresponda ao \
conteúdo do documento.

O documento está em formato bruto (pode conter marcações de tabela, descrições de \
figura/imagem e outras estruturas), mas a resposta que você receberá não terá essas \
marcações; por isso, pergunte sobre a informação/assunto, nunca sobre o formato ou as \
marcações do documento.

Perfil de quem pergunta:
{persona_instruction}

Regras:
- Escreva apenas a pergunta final, sem explicações adicionais.
- A pergunta deve ser natural e ancorada no conteúdo do documento; deixe o perfil de quem pergunta definir o estilo, o vocabulário, a extensão e o foco da pergunta.
- Não se refira às marcações nem ao formato do documento.
- Não invente fatos, cenários ou requisitos que não estejam sustentados pelo conteúdo.
- Não use referências meta como "o texto fornecido", "o trecho", "o chunk" ou números de chunk; ao citar a fonte, refira-se ao documento pelo nome.
- Responda em português."""


DOG_INSTRUCT_QUESTION_USER_PROMPT = """\
{doc_summary}Documento (formato bruto):
{sample_text}

Escreva uma única pergunta de usuário cuja resposta seja o conteúdo geral deste documento."""


DOG_INSTRUCT_REWRITE_SYSTEM_PROMPT = """\
Você é um assistente especializado. Você recebe uma pergunta de usuário e um documento \
que é a resposta a essa pergunta. O documento está em formato bruto e pouco acessível \
(pode conter marcações de tabela, descrições de figura/imagem e outras estruturas). Sua \
tarefa é reescrevê-lo levemente para que soe como a resposta natural de um assistente e \
se ajuste à pergunta, sem perder a informação.

Regras:
- O documento é a resposta: mantenha a informação e ajuste apenas a estrutura e a apresentação para ficar natural e acomodar a pergunta (acrescentando markdown, conectando partes do documento para ajudar coesão).
- Pode renderizar uma tabela de forma legível e, quando relevante, destacar valores específicos dela. Use as descrições de figura/imagem quando forem relevantes.
- Não mencione as marcações do documento (tabela, figura, imagem, etc.) nem comente que está reformatando.
- Não altere, não acrescente e não explique o conteúdo factual; não invente informações.
- Remova apenas detalhes que não dizem respeito à pergunta; mantenha o resto.
- Nunca use referências meta como "com base no texto fornecido", "no trecho", "nos trechos acima", "no contexto fornecido", "no chunk" ou números de chunk; responda como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
- Escreva apenas a resposta final, em português."""


DOG_INSTRUCT_REWRITE_USER_PROMPT = """\
Pergunta do usuário:
{question}

{doc_summary}Documento que responde à pergunta (formato bruto):
{sample_text}

Reescreva o documento como um assistente responderia à pergunta, ajustando apenas a estrutura e a apresentação e mantendo a informação relevante.
Escreva apenas a resposta final."""

DOG_INSTRUCT_PERSONAS: dict[str, str] = {
    "direto_objetivo": (
        "Funcionário da CEMIG que quer a resposta mais curta e direta possível, sem rodeios. "
        "Pergunta enxuta, focada em uma informação pontual."
    ),
    "detalhista": (
        "Funcionário da CEMIG que quer entender o contexto completo por trás da norma, não só o "
        "dado pontual. A pergunta pede justificativa, contexto ou motivo."
    ),
    "tecnico_experiente": (
        "Engenheiro ou técnico experiente que usa naturalmente a nomenclatura e as siglas das "
        "normas (ND, CE, RDP, etc.) na própria pergunta, com jargão técnico e códigos de norma."
    ),
    "leigo_area": (
        "Funcionário de outra área (ex.: administrativo) que não conhece os termos técnicos e "
        "pergunta em linguagem cotidiana, evitando jargão e usando termos genéricos ou aproximados."
    ),
    "comparador_normas": (
        "Funcionário que precisa comparar dois documentos ou assuntos técnicos para decidir qual se "
        "aplica ao caso dele. A pergunta menciona mais de um documento/tema e pede a diferença entre eles."
    ),
    "pergunta_incompleta": (
        "Funcionário que digita rápido e deixa informação implícita. A pergunta omite algum dado que "
        "normalmente seria necessário (ex.: não especifica o tipo de estrutura ou a norma)."
    ),
    "aplicacao_campo": (
        "Técnico de campo que quer saber como aplicar a norma na prática, durante uma instalação ou "
        "manutenção real. A pergunta é voltada à execução prática, não à teoria da norma."
    ),
    "confunde_documentos": (
        "Funcionário que confunde o nome ou número de normas parecidas e cita um documento próximo, "
        "mas não exatamente o correto, na pergunta."
    ),
    "urgente": (
        "Funcionário que precisa da informação rapidamente para resolver um problema em andamento. "
        "Pergunta curta, com tom de urgência, focada em resolver algo específico agora."
    ),
    "verificador_fonte": (
        "Funcionário que quer confirmar a informação e pede explicitamente a norma, item ou trecho "
        "exato que embasa a resposta, não só a informação em si."
    ),
    "pergunta_multipla": (
        "Funcionário que aproveita a mesma mensagem para fazer duas ou três perguntas relacionadas ao "
        "mesmo tema, conectadas pelo mesmo assunto."
    ),
    "confirmacao": (
        "Funcionário que já tem uma suposição em mente e só quer confirmar se está certa. Pergunta "
        'formulada como confirmação ("está correto que...", "é verdade que..."), não como pergunta aberta.'
    ),
    "gestor_resumo": (
        "Gestor que precisa de um resumo objetivo de uma norma para repassar à equipe ou usar em um "
        "relatório. A pergunta pede um resumo ou visão geral, não o detalhe técnico completo."
    ),
    "informal": (
        "Funcionário que escreve de forma informal, como numa mensagem de chat corrido, com tom "
        "coloquial e direto, sem a formalidade de um texto técnico."
    ),
    "ja_sabe_o_documento": (
        "Funcionário que já sabe o número ou nome exato da norma e pergunta diretamente sobre o "
        "conteúdo dela, citando o código do documento sem precisar descrevê-lo pelo assunto."
    ),
    "compara_versao_anterior": (
        "Funcionário que quer saber se algo mudou em relação a uma versão anterior de uma norma ou "
        "procedimento. A pergunta menciona mudança, atualização ou versão anterior do documento."
    ),
    "resposta_para_repasse": (
        "Funcionário que vai repassar a resposta diretamente a outra pessoa e por isso quer um texto "
        "pronto para reenviar como está. A pergunta pede algo 'pronto para copiar e colar', 'sem "
        "enrolação' ou 'direto', deixando claro que o texto será reutilizado sem edição."
    ),
}


# QA_LOCAL_MULTIHOP
# Context spans multiple adjacent passages joined in {sample_text}.
QA_LOCAL_MULTIHOP_PROMPT = """\
Tarefa:
Leia os trechos adjacentes abaixo, faça uma pergunta e responda a ela.

Contexto:
Os trechos são partes adjacentes do mesmo documento. A pergunta deve aproveitar a continuidade entre eles, em vez de tratar cada trecho como isolado.

Regras gerais:
1. A pergunta deve exigir o uso de pelo menos dois dos trechos fornecidos.
2. Sempre que o texto permitir, a resposta deve integrar informações de todos os trechos.
3. Não invente informações que não estejam sustentadas pelos trechos ou pelo sumário do documento.
4. Se algum trecho não for necessário para responder com rigor, a resposta pode dizer isso claramente.
5. Foque em informações factuais e importantes.
6. A resposta deve ser clara, completa e útil como resposta de um assistente.
7. Use texto simples e linguagem acessível.
8. Não use referências meta na pergunta nem na resposta (por exemplo "o texto fornecido", "os trechos acima", "trecho 1/2", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
9. Responda em português.
10. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Instruções de estilo:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado, focando no texto e só usando o sumário do texto como referencia/complemento."""


# QA_SIMILARITY_MULTIHOP
# POSITIVE: high-similarity (above) chunk groups.
QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE = """\
Tarefa:
Leia os trechos relacionados abaixo, faça uma pergunta e responda a ela.

Contexto:
Os trechos não são adjacentes. Eles foram agrupados porque possuem semelhanças, relações temáticas ou informações potencialmente complementares. Eles podem vir do mesmo documento ou de documentos diferentes.

Regras gerais:
1. A pergunta deve exigir o uso de mais de um trecho.
2. A pergunta deve aproveitar a relação entre os trechos, como semelhanças, diferenças, complementaridade, continuidade temática ou aplicação conjunta.
3. Não force uma relação inexistente: se a conexão entre os trechos for limitada, a resposta deve reconhecer esse limite.
4. Não invente informações que não estejam sustentadas pelos trechos ou pelo sumário do documento.
5. Foque em informações factuais e importantes.
6. A resposta deve ser clara, completa e útil como resposta de um assistente.
7. Use texto simples e linguagem acessível.
8. Não use referências meta na pergunta nem na resposta (por exemplo "o texto fornecido", "os trechos acima", "trecho 1/2", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
9. Responda em português.
10. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Instruções de estilo:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado, focando no texto e só usando o sumário do texto como referencia/complemento."""


# NEGATIVE: low-similarity (below / range) chunk groups for negative samples.
QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE = """\
Tarefa:
Leia os trechos pouco relacionados abaixo, faça uma pergunta e responda a ela.

Contexto:
Os trechos não são adjacentes. Eles foram agrupados porque têm baixa similaridade ou porque podem tratar de assuntos, escopos, documentos ou aplicações diferentes. O objetivo é gerar exemplos em que a resposta saiba distinguir limites, diferenças de escopo ou falta de relação suficiente entre os trechos.

Regras gerais:
1. A pergunta deve envolver mais de um trecho, mas não deve fingir que eles são fortemente relacionados se não forem.
2. A resposta deve explicar com clareza se os trechos tratam de assuntos diferentes, escopos diferentes ou informações insuficientes para uma conclusão conjunta.
3. Se houver alguma comparação válida, faça a comparação com cuidado e sem exagerar a relação.
4. Se não houver base suficiente para comparar ou combinar os trechos, diga isso explicitamente.
5. Não invente informações que não estejam sustentadas pelos trechos ou pelo sumário do documento.
6. Foque em informações factuais e importantes.
7. A resposta deve ser clara, completa e útil como resposta de um assistente.
8. Use texto simples e linguagem acessível.
9. Não use referências meta na pergunta nem na resposta (por exemplo "o texto fornecido", "os trechos acima", "trecho 1/2", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
10. Responda em português.
11. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Instruções de estilo:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado, focando no texto e só usando o sumário do texto como referencia/complemento."""


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
        'relacionar várias partes do texto, explicar um "porquê" ou um mecanismo implícito, '
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
    "applied_context": (
        "Faça uma pergunta em forma de situação prática ou dúvida realista de usuário. "
        "A pergunta deve usar o texto como base, mas sem inventar dados, condições, cenários "
        "técnicos ou requisitos que não estejam sustentados pelo conteúdo, focando no texto "
        "dado e usando o sumário apenas como apoio e forma de entender o contexto. Se o texto não tiver "
        "base suficiente para uma situação prática, formule a pergunta da forma prática mais "
        "simples possível sem extrapolar."
    ),
}


# ---------------------------------------------------------------------------
# rewrite_gen — single-turn rewrite/summarization generation.
# Two calls per (chunk × style): a user turn that writes ONLY the request, and
# an assistant turn that answers it as internal knowledge (no meta-references).
# ---------------------------------------------------------------------------

# The "instruction" IS the style: summary / simplify / focus.  Each adapts the
# user-turn persona to a different transformation of the chunk.
REWRITE_GEN_STYLE_INSTRUCTIONS: dict[str, str] = {
    "summary": (
        "Peça um resumo dos pontos principais desta seção do documento, "
        "sintetizando o conteúdo de forma fiel, sem adicionar informação externa."
    ),
    "simplify": (
        "Peça que esta seção do documento seja reescrita em linguagem mais clara "
        "e menos técnica, voltada a um leitor leigo, mantendo todos os fatos e "
        "termos técnicos essenciais e preservando o tom formal."
    ),
    "focus": (
        "Peça que um aspecto ou parte específica desta seção do documento seja "
        "reescrito ou reformulado, deixando claro qual aspecto deve ser tratado."
    ),
}

# Immutable persona for the USER turn — outputs ONLY the request.  The style
# instruction (the mutable part) is injected in the user message.
REWRITE_GEN_USER_TURN_SYSTEM_PROMPT = """\
Você é um profissional que escreve um único pedido para um assistente \
especializado nas normas técnicas da CEMIG. O pedido solicita uma transformação \
do conteúdo de uma seção específica de um documento (resumo, reescrita ou \
reformulação), conforme a instrução.

Regras invioláveis:
- Sua saída deve conter APENAS o pedido — nenhuma saudação, explicação, resposta \
ou texto adicional.
- Escreva exatamente um pedido.
- Mencione o documento pelo nome e identifique a seção em questão de forma \
natural, a partir do próprio conteúdo (ex.: "Resuma os pontos principais da \
seção de referências normativas do documento ND-4.15").
- NUNCA use referências meta como "com base no texto fornecido", "no trecho \
acima" ou números de chunk.
- Escreva em português."""

# USER-turn user message (mutable).  Fields: style_instruction, sample_text.
REWRITE_GEN_USER_TURN_USER_MSG = """\
Tipo de pedido:
{style_instruction}

{sample_text}

Escreva apenas o pedido do profissional:"""

# Immutable persona for the ASSISTANT turn.
REWRITE_GEN_ASSISTANT_SYSTEM_PROMPT = """\
Você é um assistente especializado nas normas técnicas da CEMIG.

Regras invioláveis:
- Atenda ao pedido usando o conteúdo do documento como seu próprio conhecimento.
- Nunca use referências meta como "com base no texto fornecido", "no trecho \
fornecido", "nos trechos acima", "no contexto fornecido" ou números de chunk. \
Quando precisar citar a fonte, refira-se ao documento pelo nome.
- Escreva em português."""

# ASSISTANT-turn user message (mutable).  Fields: sample_text, request,
# refusal_string.
REWRITE_GEN_ASSISTANT_USER_MSG = """\
{sample_text}

Pedido do usuário:
{request}

Atenda ao pedido usando as informações do documento acima.
- Se não houver informação suficiente para atender ao pedido, responda \
exatamente: {refusal_string}

Resposta:"""
