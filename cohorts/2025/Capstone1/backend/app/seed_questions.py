"""
Seed script for populating 200 AI Engineering questions.
Run with: python -m app.seed_questions
"""
from sqlalchemy.orm import Session
from .database import engine, Base, Question, Difficulty, Category, SessionLocal

# Create tables
Base.metadata.create_all(bind=engine)


def get_questions():
    """Returns list of 200 AI Engineering questions across categories and difficulties."""
    questions = []
    
    # ===== PYTHON - EASY (10 questions) =====
    questions.extend([
        {"text": "What is the output of print(type([]))?", "a": "<class 'list'>", "b": "<class 'array'>", "c": "<class 'tuple'>", "d": "list", "correct": "a", "diff": "easy", "cat": "python"},
        {"text": "Which keyword is used to define a function in Python?", "a": "function", "b": "def", "c": "func", "d": "define", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "What does len([1, 2, 3]) return?", "a": "2", "b": "3", "c": "4", "d": "1", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "How do you create a dictionary in Python?", "a": "dict = []", "b": "dict = ()", "c": "dict = {}", "d": "dict = <>", "correct": "c", "diff": "easy", "cat": "python"},
        {"text": "What is the correct way to import NumPy?", "a": "import numpy", "b": "include numpy", "c": "using numpy", "d": "require numpy", "correct": "a", "diff": "easy", "cat": "python"},
        {"text": "Which operator is used for exponentiation in Python?", "a": "^", "b": "**", "c": "^^", "d": "exp", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "What is the output of bool(0)?", "a": "True", "b": "False", "c": "0", "d": "None", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "How do you add an element to the end of a list?", "a": "list.add()", "b": "list.append()", "c": "list.insert()", "d": "list.push()", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "What is a list comprehension?", "a": "A way to copy lists", "b": "A concise way to create lists", "c": "A list sorting method", "d": "A list deletion method", "correct": "b", "diff": "easy", "cat": "python"},
        {"text": "What does the 'pass' keyword do?", "a": "Exits the program", "b": "Skips an iteration", "c": "Does nothing (placeholder)", "d": "Passes a value", "correct": "c", "diff": "easy", "cat": "python"},
    ])
    
    # ===== PYTHON - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is the difference between a list and a tuple?", "a": "Lists are immutable", "b": "Tuples are mutable", "c": "Tuples are immutable", "d": "No difference", "correct": "c", "diff": "medium", "cat": "python"},
        {"text": "What is a lambda function?", "a": "A named function", "b": "An anonymous function", "c": "A class method", "d": "A module", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What does *args do in a function definition?", "a": "Accepts keyword arguments", "b": "Accepts variable positional arguments", "c": "Multiplies arguments", "d": "Unpacks a dictionary", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What is the purpose of __init__ in a class?", "a": "Destructor", "b": "Constructor", "c": "Static method", "d": "Class method", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What is a generator in Python?", "a": "A function that returns a list", "b": "A function that yields values lazily", "c": "A class decorator", "d": "A type of loop", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What does the 'with' statement do?", "a": "Creates a loop", "b": "Handles context management", "c": "Defines a class", "d": "Imports modules", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What is the GIL in Python?", "a": "Global Import Lock", "b": "Global Interpreter Lock", "c": "General Interface Library", "d": "Graphics Integration Layer", "correct": "b", "diff": "medium", "cat": "python"},
        {"text": "What is the difference between == and 'is'?", "a": "No difference", "b": "== checks value, is checks identity", "c": "is checks value, == checks identity", "d": "Both check identity", "correct": "b", "diff": "medium", "cat": "python"},
    ])
    
    # ===== PYTHON - HARD (7 questions) =====
    questions.extend([
        {"text": "What is a metaclass in Python?", "a": "A class that inherits from object", "b": "A class whose instances are classes", "c": "A deprecated feature", "d": "A type of decorator", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What is the purpose of __slots__?", "a": "Define class methods", "b": "Restrict attribute creation and save memory", "c": "Create class variables", "d": "Enable multiple inheritance", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What is a descriptor in Python?", "a": "A documentation string", "b": "An object with __get__, __set__, or __delete__", "c": "A type annotation", "d": "A module attribute", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What does @functools.lru_cache do?", "a": "Logs function calls", "b": "Caches function results", "c": "Limits recursion", "d": "Measures execution time", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What is the MRO in Python?", "a": "Module Resolution Order", "b": "Method Resolution Order", "c": "Memory Reference Object", "d": "Metaclass Registration Order", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What is cooperative multiple inheritance?", "a": "Using mixins", "b": "Using super() to call parent methods in MRO", "c": "Inheriting from one class", "d": "Using interfaces", "correct": "b", "diff": "hard", "cat": "python"},
        {"text": "What is the difference between __new__ and __init__?", "a": "No difference", "b": "__new__ creates instance, __init__ initializes it", "c": "__init__ creates instance, __new__ initializes it", "d": "__new__ is deprecated", "correct": "b", "diff": "hard", "cat": "python"},
    ])
    
    # ===== ML FUNDAMENTALS - EASY (10 questions) =====
    questions.extend([
        {"text": "What is supervised learning?", "a": "Learning without labels", "b": "Learning with labeled data", "c": "Reinforcement learning", "d": "Unsupervised clustering", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is overfitting?", "a": "Model performs well on training data but poorly on new data", "b": "Model performs poorly on all data", "c": "Model is too simple", "d": "Model trains too fast", "correct": "a", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is a feature in machine learning?", "a": "The target variable", "b": "An input variable used for prediction", "c": "The model output", "d": "The loss function", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is the purpose of a validation set?", "a": "Train the model", "b": "Tune hyperparameters", "c": "Test final performance", "d": "Store data", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is linear regression used for?", "a": "Classification", "b": "Predicting continuous values", "c": "Clustering", "d": "Dimensionality reduction", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What does accuracy measure?", "a": "Percentage of correct predictions", "b": "Speed of training", "c": "Model complexity", "d": "Feature importance", "correct": "a", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is a decision tree?", "a": "A neural network", "b": "A tree-structured model for decisions", "c": "A clustering algorithm", "d": "A dimensionality reduction technique", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is the goal of unsupervised learning?", "a": "Predict labels", "b": "Find patterns without labels", "c": "Maximize rewards", "d": "Minimize training time", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is K-Means used for?", "a": "Regression", "b": "Classification", "c": "Clustering", "d": "Feature selection", "correct": "c", "diff": "easy", "cat": "ml_fundamentals"},
        {"text": "What is a hyperparameter?", "a": "A parameter learned during training", "b": "A parameter set before training", "c": "The model output", "d": "The training data", "correct": "b", "diff": "easy", "cat": "ml_fundamentals"},
    ])
    
    # ===== ML FUNDAMENTALS - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is the bias-variance tradeoff?", "a": "Choosing between accuracy and speed", "b": "Balancing underfitting (high bias) and overfitting (high variance)", "c": "Trading features for samples", "d": "Choosing between models", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is cross-validation?", "a": "Training on all data", "b": "Evaluating model on multiple train-test splits", "c": "Validating data quality", "d": "Testing in production", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is regularization?", "a": "Normalizing data", "b": "Adding penalty to prevent overfitting", "c": "Removing outliers", "d": "Feature scaling", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is the difference between L1 and L2 regularization?", "a": "L1 uses squared weights, L2 uses absolute", "b": "L1 uses absolute weights, L2 uses squared", "c": "No difference", "d": "L1 is for regression, L2 for classification", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is an ensemble method?", "a": "A single strong model", "b": "Combining multiple models for better performance", "c": "A data preprocessing step", "d": "A feature engineering technique", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is gradient descent?", "a": "A feature selection method", "b": "An optimization algorithm to minimize loss", "c": "A data augmentation technique", "d": "A model architecture", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is the purpose of the learning rate?", "a": "Determine model complexity", "b": "Control step size in gradient descent", "c": "Set number of epochs", "d": "Define batch size", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
        {"text": "What is precision in classification?", "a": "True positives / (True positives + False negatives)", "b": "True positives / (True positives + False positives)", "c": "Correct predictions / Total predictions", "d": "True negatives / Total negatives", "correct": "b", "diff": "medium", "cat": "ml_fundamentals"},
    ])
    
    # ===== ML FUNDAMENTALS - HARD (7 questions) =====
    questions.extend([
        {"text": "What is the VC dimension?", "a": "Variance-Covariance dimension", "b": "A measure of model capacity", "c": "Vector Computation dimension", "d": "Validation Count dimension", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is the PAC learning framework?", "a": "Parallel Algorithmic Computing", "b": "Probably Approximately Correct learning", "c": "Progressive Adaptive Classification", "d": "Predictive Analysis Computing", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is the kernel trick?", "a": "A data preprocessing technique", "b": "Computing dot products in high-dimensional space implicitly", "c": "A regularization method", "d": "A neural network optimization", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is the difference between bagging and boosting?", "a": "Bagging trains sequentially, boosting in parallel", "b": "Bagging trains in parallel, boosting sequentially focuses on errors", "c": "No difference", "d": "Bagging is for regression, boosting for classification", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is SHAP in explainable ML?", "a": "Shape analysis", "b": "SHapley Additive exPlanations", "c": "Statistical Hypothesis Analysis Protocol", "d": "Supervised Hyperparameter Adjustment Process", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is the curse of dimensionality?", "a": "Too few features", "b": "Data becomes sparse as dimensions increase", "c": "Model becomes too complex", "d": "Training becomes faster", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
        {"text": "What is an ROC curve?", "a": "A plot of precision vs recall", "b": "A plot of TPR vs FPR at various thresholds", "c": "A learning curve", "d": "A loss curve", "correct": "b", "diff": "hard", "cat": "ml_fundamentals"},
    ])
    
    # ===== DEEP LEARNING - EASY (8 questions) =====
    questions.extend([
        {"text": "What is a neural network?", "a": "A type of database", "b": "A model inspired by the brain with connected nodes", "c": "A programming language", "d": "A hardware device", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is an activation function?", "a": "A loss function", "b": "A function that introduces non-linearity", "c": "An optimizer", "d": "A regularization technique", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is ReLU?", "a": "A recurrent layer", "b": "Rectified Linear Unit activation", "c": "A loss function", "d": "A normalization technique", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is backpropagation?", "a": "Forward pass of data", "b": "Algorithm to compute gradients via chain rule", "c": "Data augmentation", "d": "Model initialization", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is an epoch?", "a": "One batch of training", "b": "One complete pass through the training data", "c": "A layer in the network", "d": "A hyperparameter", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is a CNN primarily used for?", "a": "Text processing", "b": "Image processing", "c": "Time series", "d": "Tabular data", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is dropout?", "a": "Removing data points", "b": "Randomly setting neurons to zero during training", "c": "Reducing learning rate", "d": "Early stopping", "correct": "b", "diff": "easy", "cat": "deep_learning"},
        {"text": "What is the purpose of pooling in CNNs?", "a": "Add more features", "b": "Reduce spatial dimensions", "c": "Increase depth", "d": "Add non-linearity", "correct": "b", "diff": "easy", "cat": "deep_learning"},
    ])
    
    # ===== DEEP LEARNING - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is batch normalization?", "a": "Normalizing input data", "b": "Normalizing layer inputs during training", "c": "Reducing batch size", "d": "A type of pooling", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is the vanishing gradient problem?", "a": "Gradients become too large", "b": "Gradients become too small in deep networks", "c": "Loss becomes zero", "d": "Model converges too fast", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is transfer learning?", "a": "Training from scratch", "b": "Using a pre-trained model for a new task", "c": "Transferring data between datasets", "d": "Moving models between GPUs", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is an LSTM?", "a": "A CNN variant", "b": "Long Short-Term Memory network for sequences", "c": "A loss function", "d": "A normalization layer", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is the purpose of the softmax function?", "a": "Add non-linearity", "b": "Convert outputs to probabilities", "c": "Regularize weights", "d": "Initialize weights", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is a residual connection (skip connection)?", "a": "Skipping layers during inference", "b": "Adding input of a block to its output", "c": "Removing layers", "d": "A type of dropout", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is the difference between SGD and Adam?", "a": "SGD uses momentum, Adam doesn't", "b": "Adam uses adaptive learning rates", "c": "No difference", "d": "SGD is faster", "correct": "b", "diff": "medium", "cat": "deep_learning"},
        {"text": "What is weight initialization important for?", "a": "Reducing model size", "b": "Ensuring stable gradients at start", "c": "Increasing accuracy", "d": "Speeding up inference", "correct": "b", "diff": "medium", "cat": "deep_learning"},
    ])
    
    # ===== DEEP LEARNING - HARD (9 questions) =====
    questions.extend([
        {"text": "What is the attention mechanism?", "a": "A regularization technique", "b": "Learning to focus on relevant parts of input", "c": "A type of loss function", "d": "A pooling operation", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is the difference between self-attention and cross-attention?", "a": "Self-attention uses same sequence for Q,K,V; cross uses different", "b": "No difference", "c": "Cross-attention is faster", "d": "Self-attention is only for images", "correct": "a", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is positional encoding in Transformers?", "a": "Encoding image positions", "b": "Adding position information to embeddings", "c": "A type of normalization", "d": "Reducing sequence length", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is knowledge distillation?", "a": "Extracting features", "b": "Training a smaller model to mimic a larger one", "c": "Data augmentation", "d": "Model pruning", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is the difference between model pruning and quantization?", "a": "Pruning reduces precision, quantization removes weights", "b": "Pruning removes weights, quantization reduces precision", "c": "No difference", "d": "Both increase model size", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is gradient checkpointing?", "a": "Saving model checkpoints", "b": "Trading compute for memory by recomputing activations", "c": "Clipping gradients", "d": "A type of optimizer", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is mixed precision training?", "a": "Using different batch sizes", "b": "Using FP16 and FP32 together for faster training", "c": "Training on multiple datasets", "d": "Using multiple optimizers", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is the Lottery Ticket Hypothesis?", "a": "Random initialization is optimal", "b": "Sparse subnetworks can match dense network performance", "c": "Deeper networks are always better", "d": "Wider networks are always better", "correct": "b", "diff": "hard", "cat": "deep_learning"},
        {"text": "What is Neural Architecture Search (NAS)?", "a": "Manual network design", "b": "Automated search for optimal architectures", "c": "Hyperparameter tuning", "d": "Data preprocessing", "correct": "b", "diff": "hard", "cat": "deep_learning"},
    ])
    
    # ===== NLP - EASY (8 questions) =====
    questions.extend([
        {"text": "What is tokenization?", "a": "Encrypting text", "b": "Splitting text into smaller units", "c": "Translating text", "d": "Compressing text", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is a word embedding?", "a": "A word dictionary", "b": "A dense vector representation of words", "c": "A grammar rule", "d": "A text file", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is sentiment analysis?", "a": "Translating languages", "b": "Determining emotional tone of text", "c": "Summarizing documents", "d": "Extracting keywords", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is NER (Named Entity Recognition)?", "a": "Network Error Resolution", "b": "Identifying entities like names, places in text", "c": "Noise Elimination Routine", "d": "Neural Entity Regression", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is TF-IDF?", "a": "A neural network", "b": "Term Frequency-Inverse Document Frequency weighting", "c": "A translation model", "d": "A tokenization method", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is a stop word?", "a": "An error in text", "b": "A common word often filtered out", "c": "A punctuation mark", "d": "A sentence ending", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is lemmatization?", "a": "Adding words", "b": "Reducing words to their base form", "c": "Counting words", "d": "Translating words", "correct": "b", "diff": "easy", "cat": "nlp"},
        {"text": "What is the bag-of-words model?", "a": "A neural network", "b": "Representing text as word frequency vectors", "c": "A translation algorithm", "d": "A compression technique", "correct": "b", "diff": "easy", "cat": "nlp"},
    ])
    
    # ===== NLP - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is Word2Vec?", "a": "A rule-based NLP system", "b": "A neural network for learning word embeddings", "c": "A translation service", "d": "A text editor", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is the difference between CBOW and Skip-gram?", "a": "CBOW predicts context from word, Skip-gram predicts word from context", "b": "CBOW predicts word from context, Skip-gram predicts context from word", "c": "No difference", "d": "CBOW is for images", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is sequence-to-sequence learning?", "a": "Binary classification", "b": "Mapping input sequence to output sequence", "c": "Feature extraction", "d": "Data augmentation", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is BLEU score used for?", "a": "Image quality", "b": "Evaluating machine translation", "c": "Audio processing", "d": "Model speed", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is beam search?", "a": "A training algorithm", "b": "A decoding strategy that keeps top-k hypotheses", "c": "A loss function", "d": "A data structure", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is subword tokenization?", "a": "Character-level only", "b": "Breaking words into frequent subword units", "c": "Sentence-level tokenization", "d": "Document-level tokenization", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is perplexity in language modeling?", "a": "Model accuracy", "b": "Measure of how well model predicts text", "c": "Training speed", "d": "Vocabulary size", "correct": "b", "diff": "medium", "cat": "nlp"},
        {"text": "What is coreference resolution?", "a": "Fixing typos", "b": "Identifying expressions referring to same entity", "c": "Sentence splitting", "d": "Word counting", "correct": "b", "diff": "medium", "cat": "nlp"},
    ])
    
    # ===== NLP - HARD (9 questions) =====
    questions.extend([
        {"text": "What is the Transformer architecture based on?", "a": "RNNs", "b": "Self-attention without recurrence", "c": "CNNs", "d": "Random forests", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is masked language modeling (MLM)?", "a": "Predicting next word", "b": "Predicting masked tokens in bidirectional context", "c": "Translation", "d": "Summarization", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is the difference between BERT and GPT?", "a": "BERT is autoregressive, GPT is bidirectional", "b": "BERT is bidirectional, GPT is autoregressive", "c": "No difference", "d": "BERT is for images", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is constitutional AI?", "a": "A government AI system", "b": "Training AI with principles/rules for safety", "c": "A legal document parser", "d": "A voting algorithm", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is RLHF?", "a": "Recursive Learning Hierarchical Framework", "b": "Reinforcement Learning from Human Feedback", "c": "Random Layer Hyperparameter Fixing", "d": "Regularized Linear Hypothesis Fitting", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is the purpose of the [CLS] token in BERT?", "a": "End of sentence", "b": "Aggregate sequence representation", "c": "Unknown word", "d": "Padding", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is contrastive learning in NLP?", "a": "Comparing models", "b": "Learning by comparing similar and dissimilar examples", "c": "A loss function", "d": "A tokenization method", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is prefix tuning?", "a": "Fine-tuning all parameters", "b": "Optimizing continuous prompts prepended to input", "c": "Data preprocessing", "d": "Model pruning", "correct": "b", "diff": "hard", "cat": "nlp"},
        {"text": "What is the difference between hard and soft prompts?", "a": "Hard prompts are better", "b": "Hard are discrete tokens, soft are learned embeddings", "c": "Soft prompts are discrete", "d": "No difference", "correct": "b", "diff": "hard", "cat": "nlp"},
    ])
    
    # ===== MLOPS - EASY (8 questions) =====
    questions.extend([
        {"text": "What is MLOps?", "a": "Machine Learning Operations - practices for ML lifecycle", "b": "A programming language", "c": "A database", "d": "A hardware device", "correct": "a", "diff": "easy", "cat": "mlops"},
        {"text": "What is a ML pipeline?", "a": "A data cable", "b": "A sequence of automated ML steps", "c": "A model architecture", "d": "A loss function", "correct": "b", "diff": "easy", "cat": "mlops"},
        {"text": "What is model versioning?", "a": "Updating model code", "b": "Tracking different versions of trained models", "c": "Renaming models", "d": "Deleting old models", "correct": "b", "diff": "easy", "cat": "mlops"},
        {"text": "What is containerization in ML?", "a": "Storing data", "b": "Packaging applications with dependencies (e.g., Docker)", "c": "Compressing models", "d": "Encrypting data", "correct": "b", "diff": "easy", "cat": "mlops"},
        {"text": "What is CI/CD in ML?", "a": "Continuous Integration/Deployment for ML systems", "b": "A model architecture", "c": "A dataset format", "d": "A cloud provider", "correct": "a", "diff": "easy", "cat": "mlops"},
        {"text": "What is model monitoring?", "a": "Watching training", "b": "Tracking model performance in production", "c": "Debugging code", "d": "Testing locally", "correct": "b", "diff": "easy", "cat": "mlops"},
        {"text": "What is an ML registry?", "a": "A model store", "b": "Central repository for managing ML artifacts", "c": "A training script", "d": "A deployment tool", "correct": "b", "diff": "easy", "cat": "mlops"},
        {"text": "What is feature engineering?", "a": "Building hardware", "b": "Creating informative features from raw data", "c": "Deploying models", "d": "Monitoring systems", "correct": "b", "diff": "easy", "cat": "mlops"},
    ])
    
    # ===== MLOPS - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is data drift?", "a": "Data storage issue", "b": "Change in data distribution over time affecting model", "c": "Data corruption", "d": "Data loss", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is concept drift?", "a": "Change in data storage", "b": "Change in relationship between input and target", "c": "Model size change", "d": "Feature change", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is A/B testing in ML?", "a": "Testing two models concurrently to compare performance", "b": "Training two models", "c": "Splitting data", "d": "Debugging models", "correct": "a", "diff": "medium", "cat": "mlops"},
        {"text": "What is a feature store?", "a": "A code repository", "b": "Centralized repository for storing and serving features", "c": "A model registry", "d": "A deployment platform", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is model serving?", "a": "Training models", "b": "Making models available for inference", "c": "Storing models", "d": "Deleting models", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is blue-green deployment?", "a": "Color-coded models", "b": "Running two production environments for zero-downtime updates", "c": "A testing strategy", "d": "A monitoring tool", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is canary deployment?", "a": "Deploying to all users", "b": "Gradually rolling out to a subset of users", "c": "Testing in development", "d": "Rollback strategy", "correct": "b", "diff": "medium", "cat": "mlops"},
        {"text": "What is experiment tracking?", "a": "Logging model training runs and results", "b": "Testing hypotheses", "c": "Data versioning", "d": "Code review", "correct": "a", "diff": "medium", "cat": "mlops"},
    ])
    
    # ===== MLOPS - HARD (9 questions) =====
    questions.extend([
        {"text": "What is ML lineage tracking?", "a": "Tracking model ancestors", "b": "Recording data, code, and model relationships", "c": "Version control", "d": "Monitoring uptime", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is shadow mode deployment?", "a": "Dark deployment", "b": "Running new model alongside production without affecting output", "c": "Encrypted deployment", "d": "Private deployment", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is the purpose of model cards?", "a": "Credit cards for ML", "b": "Documentation of model details, usage, and limitations", "c": "Model identification", "d": "Hardware specifications", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is infrastructure as code (IaC) in MLOps?", "a": "Writing ML code", "b": "Managing infrastructure through code/config files", "c": "Database management", "d": "Model training", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is the difference between online and batch inference?", "a": "Online is slower", "b": "Online is real-time, batch processes data in bulk", "c": "Batch is real-time", "d": "No difference", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is model explainability in production?", "a": "Model documentation", "b": "Providing interpretable predictions to users", "c": "Code comments", "d": "Error logging", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is GitOps for ML?", "a": "Using Git for code", "b": "Using Git as single source of truth for ML deployments", "c": "GitHub for ML", "d": "Version control only", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is model governance?", "a": "Government regulations", "b": "Policies and controls for ML model lifecycle", "c": "Model training", "d": "Data collection", "correct": "b", "diff": "hard", "cat": "mlops"},
        {"text": "What is continuous training (CT)?", "a": "Training without breaks", "b": "Automatically retraining models when needed", "c": "Manual retraining", "d": "One-time training", "correct": "b", "diff": "hard", "cat": "mlops"},
    ])
    
    # ===== DATA ENGINEERING - EASY (8 questions) =====
    questions.extend([
        {"text": "What is ETL?", "a": "Extract, Transform, Load", "b": "Enter, Test, Leave", "c": "Encrypt, Transfer, Lock", "d": "Edit, Translate, Link", "correct": "a", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is a data warehouse?", "a": "A physical storage room", "b": "A central repository for structured data", "c": "A database backup", "d": "A data cable", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is SQL used for?", "a": "Machine learning", "b": "Querying relational databases", "c": "Image processing", "d": "Web development", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is a data lake?", "a": "A body of water with data", "b": "A storage repository for raw data", "c": "A cleaned database", "d": "A data visualization", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is Apache Spark?", "a": "A database", "b": "A distributed computing framework", "c": "A programming language", "d": "A cloud provider", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is a DataFrame?", "a": "A picture frame", "b": "A 2D labeled data structure", "c": "A database table only", "d": "A file format", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is data normalization?", "a": "Deleting data", "b": "Organizing data to reduce redundancy", "c": "Backing up data", "d": "Encrypting data", "correct": "b", "diff": "easy", "cat": "data_engineering"},
        {"text": "What is a primary key?", "a": "The most important data", "b": "A unique identifier for table rows", "c": "A password", "d": "A data type", "correct": "b", "diff": "easy", "cat": "data_engineering"},
    ])
    
    # ===== DATA ENGINEERING - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is the difference between OLTP and OLAP?", "a": "OLTP for analytics, OLAP for transactions", "b": "OLTP for transactions, OLAP for analytics", "c": "No difference", "d": "Both are protocols", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is data partitioning?", "a": "Splitting data across storage units for efficiency", "b": "Deleting partial data", "c": "Encrypting data", "d": "Compressing data", "correct": "a", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is a star schema?", "a": "A database shape", "b": "A data warehouse design with fact and dimension tables", "c": "A rating system", "d": "A encryption method", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is Apache Kafka used for?", "a": "Database storage", "b": "Real-time event streaming", "c": "Machine learning", "d": "Web hosting", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is data lineage?", "a": "Data ancestry", "b": "Tracking data origin and transformations", "c": "Data deletion", "d": "Data encryption", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is a materialized view?", "a": "A visible table", "b": "A pre-computed query result stored as a table", "c": "A temporary view", "d": "A user interface", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is data sharding?", "a": "Breaking data", "b": "Horizontally partitioning data across databases", "c": "Compressing data", "d": "Encrypting data", "correct": "b", "diff": "medium", "cat": "data_engineering"},
        {"text": "What is Apache Airflow?", "a": "A weather API", "b": "A workflow orchestration platform", "c": "A database", "d": "A messaging queue", "correct": "b", "diff": "medium", "cat": "data_engineering"},
    ])
    
    # ===== DATA ENGINEERING - HARD (9 questions) =====
    questions.extend([
        {"text": "What is the CAP theorem?", "a": "A data compression method", "b": "Consistency, Availability, Partition tolerance tradeoff", "c": "A programming paradigm", "d": "A security protocol", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is eventual consistency?", "a": "Immediate consistency", "b": "Data will be consistent eventually after updates propagate", "c": "Never consistent", "d": "Always consistent", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is a write-ahead log (WAL)?", "a": "A planning document", "b": "Logging changes before applying for durability", "c": "A backup system", "d": "A monitoring tool", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is data mesh?", "a": "A data cable network", "b": "Decentralized data architecture with domain ownership", "c": "A visualization tool", "d": "A database type", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is change data capture (CDC)?", "a": "Changing database schema", "b": "Tracking and capturing data changes in real-time", "c": "Data compression", "d": "Data encryption", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is a lambda architecture?", "a": "AWS Lambda only", "b": "Combining batch and real-time processing layers", "c": "A serverless function", "d": "A programming pattern", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is data quality testing?", "a": "Testing data speed", "b": "Validating data accuracy, completeness, consistency", "c": "Testing database performance", "d": "Testing user interface", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is a slowly changing dimension (SCD)?", "a": "A static dimension", "b": "Techniques to handle dimension changes over time", "c": "A fast dimension", "d": "A deleted dimension", "correct": "b", "diff": "hard", "cat": "data_engineering"},
        {"text": "What is data contracts?", "a": "Legal agreements", "b": "Agreements on data schema and quality between producers/consumers", "c": "Database licenses", "d": "Storage agreements", "correct": "b", "diff": "hard", "cat": "data_engineering"},
    ])
    
    # ===== MATH & STATS - EASY (8 questions) =====
    questions.extend([
        {"text": "What is the mean of a dataset?", "a": "The middle value", "b": "The average of all values", "c": "The most common value", "d": "The range of values", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is standard deviation?", "a": "The average value", "b": "A measure of data spread", "c": "The median", "d": "The mode", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is a probability distribution?", "a": "A data table", "b": "A function describing likelihood of outcomes", "c": "A sorting algorithm", "d": "A database query", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is a matrix?", "a": "A movie", "b": "A 2D array of numbers", "c": "A 1D list", "d": "A graph", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is the dot product?", "a": "A punctuation mark", "b": "Sum of element-wise products of two vectors", "c": "Matrix multiplication", "d": "Vector addition", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is correlation?", "a": "Causation", "b": "A measure of linear relationship between variables", "c": "Data independence", "d": "Feature importance", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is variance?", "a": "The mean", "b": "The average squared deviation from mean", "c": "The range", "d": "The median", "correct": "b", "diff": "easy", "cat": "math_stats"},
        {"text": "What is a normal distribution?", "a": "Any distribution", "b": "A bell-shaped symmetric distribution", "c": "A uniform distribution", "d": "A skewed distribution", "correct": "b", "diff": "easy", "cat": "math_stats"},
    ])
    
    # ===== MATH & STATS - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is Bayes' theorem?", "a": "A sorting algorithm", "b": "A formula for conditional probability", "c": "A matrix operation", "d": "A optimization method", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is the central limit theorem?", "a": "A data centering method", "b": "Sample means approach normal distribution as n increases", "c": "A clustering algorithm", "d": "A dimensionality reduction", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is eigenvalue decomposition?", "a": "Breaking matrices", "b": "Factorizing matrix into eigenvectors and eigenvalues", "c": "Matrix addition", "d": "Matrix inversion", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is gradient in calculus?", "a": "A slope measurement", "b": "A vector of partial derivatives", "c": "An integral", "d": "A constant", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is the chain rule?", "a": "A programming rule", "b": "Rule for differentiating composite functions", "c": "A data structure", "d": "A sorting algorithm", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is a p-value?", "a": "Probability of Type II error", "b": "Probability of observing results given null hypothesis is true", "c": "Model accuracy", "d": "Feature importance", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is maximum likelihood estimation?", "a": "Finding maximum value", "b": "Finding parameters that maximize probability of data", "c": "A clustering method", "d": "A regression type", "correct": "b", "diff": "medium", "cat": "math_stats"},
        {"text": "What is the difference between Type I and Type II errors?", "a": "Type I is false negative, Type II is false positive", "b": "Type I is false positive, Type II is false negative", "c": "No difference", "d": "Both are correct decisions", "correct": "b", "diff": "medium", "cat": "math_stats"},
    ])
    
    # ===== MATH & STATS - HARD (9 questions) =====
    questions.extend([
        {"text": "What is KL divergence?", "a": "A distance metric", "b": "A measure of difference between probability distributions", "c": "A clustering algorithm", "d": "A normalization technique", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is the Hessian matrix?", "a": "A German mathematician", "b": "Matrix of second-order partial derivatives", "c": "A data matrix", "d": "An identity matrix", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is SVD (Singular Value Decomposition)?", "a": "A data format", "b": "Factorizing matrix into U, Σ, V^T", "c": "A neural network", "d": "A loss function", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is the ELBO in variational inference?", "a": "A body part", "b": "Evidence Lower Bound for approximating posteriors", "c": "A loss function", "d": "An optimizer", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is the Fisher information?", "a": "Information about fish", "b": "A measure of information a sample provides about a parameter", "c": "A database query", "d": "A data structure", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is Markov Chain Monte Carlo (MCMC)?", "a": "A game", "b": "A method for sampling from probability distributions", "c": "A neural network", "d": "A optimization algorithm", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is the reparameterization trick?", "a": "Renaming parameters", "b": "Making sampling differentiable for backpropagation", "c": "A regularization method", "d": "A data augmentation", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is the Law of Large Numbers?", "a": "Big numbers are better", "b": "Sample average converges to expected value as n increases", "c": "A programming law", "d": "A database rule", "correct": "b", "diff": "hard", "cat": "math_stats"},
        {"text": "What is Jensen's inequality?", "a": "An equality", "b": "Relationship between convex functions and expectations", "c": "A sorting algorithm", "d": "A data structure", "correct": "b", "diff": "hard", "cat": "math_stats"},
    ])
    
    # ===== LLM - EASY (8 questions) =====
    questions.extend([
        {"text": "What does LLM stand for?", "a": "Limited Learning Model", "b": "Large Language Model", "c": "Linear Learning Machine", "d": "Local Language Model", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is a prompt in LLMs?", "a": "An error message", "b": "Input text given to the model", "c": "The model output", "d": "A training algorithm", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is ChatGPT?", "a": "A database", "b": "A conversational AI powered by GPT", "c": "A programming language", "d": "A search engine", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is token in LLMs?", "a": "A coin", "b": "A unit of text (word or subword)", "c": "A password", "d": "A user ID", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is temperature in LLM generation?", "a": "Model heating", "b": "Parameter controlling output randomness", "c": "Training speed", "d": "Model size", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is zero-shot learning?", "a": "No training", "b": "Performing tasks without task-specific examples", "c": "Perfect accuracy", "d": "No inference", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is few-shot learning?", "a": "Fast training", "b": "Learning from a few examples in the prompt", "c": "Short responses", "d": "Small models", "correct": "b", "diff": "easy", "cat": "llm"},
        {"text": "What is hallucination in LLMs?", "a": "Visual output", "b": "Generating false or fabricated information", "c": "Correct responses", "d": "Model errors", "correct": "b", "diff": "easy", "cat": "llm"},
    ])
    
    # ===== LLM - MEDIUM (8 questions) =====
    questions.extend([
        {"text": "What is prompt engineering?", "a": "Building prompts", "b": "Designing effective prompts for desired outputs", "c": "Training LLMs", "d": "Deploying models", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is RAG (Retrieval Augmented Generation)?", "a": "A cleaning method", "b": "Combining retrieval with generation for grounded responses", "c": "A training technique", "d": "A model architecture", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is fine-tuning an LLM?", "a": "Making it smaller", "b": "Training on task-specific data after pretraining", "c": "Adjusting temperature", "d": "Changing prompts", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is a context window?", "a": "A GUI element", "b": "Maximum number of tokens model can process", "c": "A training parameter", "d": "A deployment setting", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is chain-of-thought prompting?", "a": "Linking prompts", "b": "Prompting model to show reasoning steps", "c": "A training method", "d": "A model architecture", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is embedding in LLMs?", "a": "Hiding information", "b": "Vector representation of text", "c": "Model compression", "d": "Data encryption", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is top-p (nucleus) sampling?", "a": "Selecting top parameters", "b": "Sampling from tokens comprising top-p probability mass", "c": "A training method", "d": "A loss function", "correct": "b", "diff": "medium", "cat": "llm"},
        {"text": "What is instruction tuning?", "a": "Writing instructions", "b": "Fine-tuning on instruction-following data", "c": "Prompt formatting", "d": "Model deployment", "correct": "b", "diff": "medium", "cat": "llm"},
    ])
    
    # ===== LLM - HARD (9 questions) =====
    questions.extend([
        {"text": "What is LoRA (Low-Rank Adaptation)?", "a": "A model name", "b": "Parameter-efficient fine-tuning using low-rank matrices", "c": "A loss function", "d": "A data format", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is QLoRA?", "a": "A Q-learning variant", "b": "Quantized LoRA for memory-efficient fine-tuning", "c": "A model architecture", "d": "A dataset", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is the purpose of rotary position embeddings (RoPE)?", "a": "Tying embeddings", "b": "Encoding relative positions efficiently", "c": "Compressing models", "d": "Generating images", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is KV cache in LLM inference?", "a": "A storage system", "b": "Caching key-value pairs to speed up generation", "c": "A training technique", "d": "A loss function", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is speculative decoding?", "a": "Guessing outputs", "b": "Using small model to draft, large model to verify", "c": "A training method", "d": "A loss function", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is MoE (Mixture of Experts)?", "a": "Expert panel", "b": "Architecture with specialized sub-networks", "c": "A training method", "d": "A dataset", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is DPO (Direct Preference Optimization)?", "a": "A training dataset", "b": "Alternative to RLHF using preference data directly", "c": "A model architecture", "d": "A loss function only", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is Flash Attention?", "a": "Fast training", "b": "Memory-efficient attention computation", "c": "A model name", "d": "A dataset", "correct": "b", "diff": "hard", "cat": "llm"},
        {"text": "What is model merging in LLMs?", "a": "Combining datasets", "b": "Combining weights of multiple models", "c": "Training together", "d": "Deploying together", "correct": "b", "diff": "hard", "cat": "llm"},
    ])
    
    return questions


def seed_database():
    """Populate the database with questions."""
    db = SessionLocal()
    
    # Check if already seeded
    existing = db.query(Question).count()
    if existing > 0:
        print(f"Database already has {existing} questions. Skipping seed.")
        db.close()
        return
    
    questions = get_questions()
    
    diff_map = {"easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD}
    cat_map = {
        "python": Category.PYTHON,
        "ml_fundamentals": Category.ML_FUNDAMENTALS,
        "deep_learning": Category.DEEP_LEARNING,
        "nlp": Category.NLP,
        "mlops": Category.MLOPS,
        "data_engineering": Category.DATA_ENGINEERING,
        "math_stats": Category.MATH_STATS,
        "llm": Category.LLM
    }
    
    for q in questions:
        question = Question(
            text=q["text"],
            option_a=q["a"],
            option_b=q["b"],
            option_c=q["c"],
            option_d=q["d"],
            correct_option=q["correct"],
            difficulty=diff_map[q["diff"]],
            category=cat_map[q["cat"]]
        )
        db.add(question)
    
    db.commit()
    print(f"Seeded {len(questions)} questions!")
    db.close()


if __name__ == "__main__":
    seed_database()
