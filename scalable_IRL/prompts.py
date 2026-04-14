
prompts = {
    'trl-lib/tldr': {
                     'prompt': """Summarize the following post in a few sentences.

{input}""",
                     'input': 'prompt',
                     'target': 'completion'
                    },

    'EdinburghNLP/xsum': {
                     'prompt': """Summarize the following article in one sentence.

Article: {input}

Summary: """,
                     'input': 'document',
                     'target': 'summary'
    }
}
