import re
def _extract_answers_only(text: str) -> str:
    """
    Parses text to return ONLY the student's answers.
    Handles:
    - Standard list (1. Answer)
    - Wall of text (Question 1: Answer)
    - Metadata headers
    """
    clean_answers = []

    # 1. CLEANUP HEADERS
    # Find the first occurrence of "Question 1", "Q1", or "1." to discard top-level metadata
    # This prevents headers like "Student ID: 123" from confusing the parser.
    match_start = re.search(r'(?:Question|Q|Section)\s*1\s*[:\.\)\-]|(?<=^)\s*1\.|(?<=\n)\s*1\.', text, re.IGNORECASE)
    if match_start:
        text = text[match_start.start():]

    # 2. NORMALIZE QUESTION DELIMITERS (THE FIX)
    # We use a capture group (\1) to preserve the punctuation of the previous sentence.
    # Logic: Look for (Punctuation/Start) + (Number) + (Separator)
    text = re.sub(
        r'(?i)([\.\?!]\s*|\n|^)(?:Question\s*\d+|Q\.?\s*\d+|Section\s*\d+|\d+)\s*[:\.\)\-]', 
        r'\1\n__SPLIT__QUESTION__\n', 
        text
    )

    # 3. SPLIT INTO CHUNKS
    fragments = text.split('__SPLIT__QUESTION__')

    for fragment in fragments:
        if not fragment.strip():
            continue

        # 4. FIND THE ANSWER START
        # Looks for "Ans -", "Answer:", "A:", or just the text if structured explicitly
        answer_parts = re.split(
            r'(?i)(?:Ans|Answer|A)\s*[:\-\–\.]', 
            fragment, 
            maxsplit=1
        )

        if len(answer_parts) > 1:
            # Case A: Found an explicit "Ans -" marker
            raw_answer = answer_parts[1].strip()
            if raw_answer:
                clean_answers.append(raw_answer)
        else:
            # Case B: No explicit marker (e.g., "1. Machine Learning is...")
            # If we successfully split by Question Number, the whole remaining fragment is likely the answer.
            # We filter out very short junk (like empty strings)
            content = fragment.strip()
            if len(content) > 1:
                clean_answers.append(content)

    # 5. FINAL JOIN
    if not clean_answers:
        return text.strip()

    return " ".join(clean_answers)
test_text = '''STUDENT 3 - LOW RISK OF PLAGIARISM Question 1: What is the value of the probability that a model ranks a random positive example higher than a random negative example represented by the ROC curve? Answer: The ROC (Receiver Operating Characteristic) curve provides a comprehensive visualization of a binary classifier's performance across all possible decision thresholds. The area under this curve, known as AUC (Area Under the Curve), has a specific probabilistic interpretation that is very useful for understanding model quality. The AUC value represents the probability that when you randomly select one positive instance and one negative instance from your dataset, the classifier will assign a higher score to the positive instance than to the negative instance. This probability interpretation makes AUC intuitive and meaningful. A perfect classifier achieves an AUC of 1.0, meaning it always ranks positive examples higher than negative ones. A completely random classifier, which is no better than guessing, has an AUC of 0.5, indicating a 50% chance of correctly ranking a random positive- negative pair. Most practical classifiers fall between these extremes, with values typically ranging from 0.6 to 0.95. Higher AUC values indicate better discriminative ability - the model is better at distinguishing between the two classes. One significant advantage of AUC is that it's threshold- independent, meaning it evaluates the model's ranking ability across all possible thresholds rather than depending on a single chosen threshold. This makes it particularly valuable for comparing different models or when the optimal classification threshold is uncertain or may vary in different deployment scenarios. Question 2: How does prioritizing fairness and transparency contribute to building responsible Al systems? Answer: Fairness and transparency are fundamental pillars for developing responsible Al systems that benefit society while minimizing potential harms. Prioritizing fairness means ensuring that Al systems provide equitable treatment across different demographic groups and don't perpetuate or amplify existing societal biases. When Al systems are used for important decisions like loan approvals, hiring, or criminal justice, unfair algorithms can systematically disadvantage certain populations. By actively addressing fairness, developers can identify and mitigate biases in training data, implement fairness constraints during model development, and monitor deployed systems for discriminatory impacts. This requires diverse training data that represents all relevant populations, testing for disparate performance across demographic groups, and ongoing auditing of system outputs. Transparency addresses the interpretability challenge inherent in complex machine learning models. Many sophisticated models, especially deep neural networks, function as "black boxes" where it's difficult to understand why specific decisions are made. Transparency means making Al systems more explainable and understandable to stakeholders - including users, regulators, and affected individuals. This includes documenting data sources and modeling approaches, providing clear

explanations for predictions, communicating model limitations honestly, and making performance metrics accessible. Together, fairness and transparency create accountability in Al systems. They enable meaningful auditing, allow individuals to challenge automated decisions, support regulatory compliance, and build public trust. Trust is essential because without it, beneficial Al applications may face resistance. These principles also enable continuous improvement - when you can understand how a model works and measure its fairness, you can identify and address problems systematically. Responsible Al development recognizes that technical performance alone is insufficient - ethical considerations must be integrated throughout the machine learning lifecycle from problem definition through deployment and monitoring. Question 3: What is an output of Logistic Regression for classification? Answer: Logistic regression produces probability values as its primary output, which makes it well-suited for binary classification tasks. Despite having "regression" in its name, logistic regression is designed for classification. The output is a probability between O and 1 that indicates the model's estimated likelihood that an instance belongs to the positive class. This probability is generated by applying the sigmoid function (also called the logistic function) to a linear combination of input features and learned weights. The sigmoid function has the mathematical form o(z) = 1/(1 + e4(-z)), which transforms any real-valued input into the probability range (0,1). For example, in spam email detection, logistic regression might output 0.85 for a particular email, indicating 85% probability that it's spam. These probabilistic outputs are valuable because they provide both a prediction and a confidence level. To make actual classification decisions, a threshold is applied to these probabilities - typically 0.5, where probabilities above this threshold predict the positive class and those below predict the negative class. However, this threshold can be adjusted based on application requirements. In medical diagnosis where missing a disease is very costly, a lower threshold (like 0.3) might be used to increase sensitivity. For spam filtering where false positives are problematic, a higher threshold (like 0.7) might be preferred. The probability outputs also enable ranking instances by their likelihood of being positive, which is useful for prioritization tasks. For multi-class classification problems, logistic regression extends to multinomial (softmax) regression, which outputs a probability for each possible class, with all probabilities summing to 1, and the class with highest probability typically selected as the prediction. Question 4: What is the primary purpose of using traffic flow prediction in smart city applications? Answer: Traffic flow prediction serves as a key technology in smart city infrastructure, with its primary purpose being to optimize urban transportation through proactive management of traffic conditions. By using machine learning algorithms to forecast traffic patterns based on historical data, real-time sensors, weather conditions, special events, and other factors, cities can anticipate and prevent congestion before it develops. This predictive capability enables several important benefits. First, it allows for intelligent traffic signal control that can adjust timing patterns proactively based on

predicted conditions rather than just reacting to current traffic. For instance, if heavy traffic is predicted on certain routes during evening rush hour, signal timings can be optimized in advance to facilitate smoother flow. Second, traffic predictions enable better route guidance through navigation applications, helping drivers avoid anticipated congestion and distributing traffic more evenly across the road network, which reduces overall travel times and fuel consumption. Public transportation benefits as well, with transit agencies using predictions to optimize bus schedules and routes, improving service reliability and operational efficiency. From an urban planning perspective, long- term traffic predictions help identify where infrastructure investments like new roads or public transit would be most beneficial. Environmental benefits include reduced emissions from less congestion and idling. Economic impacts are significant - less time wasted in traffic improves productivity, reduces fuel costs, and decreases vehicle maintenance from stop-and-go driving. Traffic predictions also support smart parking systems that guide drivers efficiently to available spaces, and they provide context for autonomous vehicle path planning. Overall, traffic flow prediction transforms urban transportation from a reactive system into a proactive one that anticipates and mitigates problems before they fully develop, improving quality of life for city residents. Question 5: Which regularization technique is described as performing feature selection by adding penalty Az | weights | ? Answer: The regularization technique with the penalty term AZ| weights| is L1 regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator) regularization. This technique is distinctive because it performs automatic feature selection during the model training process. L1 regularization works by adding the sum of the absolute values of model weights (multiplied by a regularization parameter )) to the loss function. During optimization, the algorithm must balance minimizing prediction error with minimizing this penalty term. The key property that enables feature selection is the use of absolute values rather than squared values. The mathematical properties of the absolute value function create "corners" at zero in the optimization landscape, which causes many weights to be driven exactly to zero rather than just becoming small. When a weight becomes zero, the corresponding feature is effectively removed from the model. This automatic feature selection provides several advantages. It improves model interpretability by reducing the number of active features, which makes it easier to understand which factors drive predictions. It can enhance generalization performance by eliminating noise from irrelevant features. It also reduces computational costs since only non-zero features need to be considered during prediction. The regularization parameter A controls the strength of feature selection: larger A values impose stronger penalties, forcing more weights to zero; smaller A values allow more features to remain. The optimal A is typically found through cross-validation. L1 regularization is particularly useful in high- dimensional settings where you have many features but expect only a subset to be truly relevant, such as in genomics, text classification, or financial modeling where you might have thousands of potential features but believe only dozens are actually important for predictions.'''

# print(_extract_answers_only(test_text))