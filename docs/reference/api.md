# The lab code and helper API

The numerical, parallel, and GPU directories contain independent teaching scripts. Each script keeps its formulas close to its experiment. This makes a small example easier to inspect than a large shared implementation.

## The quiz helper renders review questions.

`labs/tutorial_helpers.py` provides `quiz(questions)` for notebooks and `quiz_html(questions)` for the Sphinx directive. Each question has four fields.

| Field | Meaning |
|---|---|
| `q` contains the question text. | The text asks about a stated learning goal. |
| `options` contains the candidate answers. | Each option is a string. |
| `answer` contains the correct option index. | The index starts at zero. |
| `explain` contains the explanation. | The explanation states why the answer follows. |

```python
question = {
    "q": "Does a small residual always imply a small forward error?",
    "options": ["Yes.", "No."],
    "answer": 1,
    "explain": "An ill-conditioned problem can amplify a small residual.",
}
```

## The verification script records execution evidence.

`python scripts/verify_course.py` checks the authored Markdown lessons and executes the portable Python labs in subprocesses. The script records return codes, output, input-source hashes, and the Python platform. A failure ends the verification with a nonzero status.

## The book builder collects the same authored lessons.

`python scripts/build_book.py` builds `dist/STUDY_GUIDE.md`. The builder expands supported literal includes, converts interactive quizzes to readable questions and answers, and preserves mathematical notation. The generated book is a reading edition. You should edit the lesson sources and rebuild it.
