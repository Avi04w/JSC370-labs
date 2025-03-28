Lab 08 - Text Mining/NLP
================

# Learning goals

- Use `unnest_tokens()` and `unnest_ngrams()` to extract tokens and
  ngrams from text
- Use dplyr and ggplot2 to analyze and visualize text data
- Try a theme model using `topicmodels`

# Lab description

For this lab we will be working with the medical record transcriptions
from <https://www.mtsamples.com/> available at
<https://github.com/JSC370/JSC370-2025/tree/main/data/medical_transcriptions>.

# Deliverables

1.  Questions 1-7 answered, knit to pdf or html output uploaded to
    Quercus.

2.  Render the Rmarkdown document using `github_document` and add it to
    your github site. Add link to github site in your html.

### Setup packages

You should load in `tidyverse`, (or `data.table`), `tidytext`,
`wordcloud2`, `tm`, and `topicmodels`.

## Read in the Medical Transcriptions

Loading in reference transcription samples from
<https://www.mtsamples.com/>

``` r
library(tidytext)
library(tidyverse)
library(wordcloud2)
library(tm)
library(topicmodels)

mt_samples <- read_csv("https://raw.githubusercontent.com/JSC370/JSC370-2025/main/data/medical_transcriptions/mtsamples.csv")
mt_samples <- mt_samples |>
  select(description, medical_specialty, transcription)

head(mt_samples)
```

------------------------------------------------------------------------

## Question 1: What specialties do we have?

We can use `count()` from `dplyr` to figure out how many different
medical specialties are in the data. Are these categories related?
overlapping? evenly distributed? Make a bar plot.

``` r
mt_samples |>
  count(medical_specialty, sort = TRUE) |>
  ggplot(aes(x = reorder(medical_specialty, n), y = n)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Distribution of Medical Specialties", x = "Medical Specialty", y = "Count")
```

Surgery has a much higher frequency than any other specialty, more than
triple the next most common one which is orthopedics.

------------------------------------------------------------------------

## Question 2: Tokenize

- Tokenize the the words in the `transcription` column
- Count the number of times each token appears
- Visualize the top 20 most frequent words with a bar plot
- Create a word cloud of the top 20 most frequent words

### Explain what we see from this result. Does it makes sense? What insights (if any) do we get?

``` r
tokens <- mt_samples |>
  unnest_tokens(token, transcription) |>
  count(token, sort=TRUE)

tokens |>
  slice_max(n, n=20) |> 
  ggplot(aes(x = reorder(token, n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Words", x = "Words", y = "Frequency")

wordcloud2(tokens[1:20, ])
```

the most common words are the, and, was, …, mostly stopwords. This makes
sense as they will be in most sentences, regardless of topic. We get no
insights from this.

------------------------------------------------------------------------

## Question 3: Stopwords

- Redo Question 2 but remove stopwords
- Check `stopwords()` library and `stop_words` in `tidytext`
- Use regex to remove numbers as well
- Try customizing your stopwords list to include 3-4 additional words
  that do not appear informative

### What do we see when you remove stopwords and then when you filter further? Does it give us a better idea of what the text is about?

``` r
head(stopwords("english"))
length(stopwords("english"))
head(stop_words)
stop_words_custom <- c(stopwords("english"), "patient", "placed", "using")

tokens <- mt_samples |>
  unnest_tokens(token, transcription) |>
  filter(!(token %in% stop_words_custom), !str_detect(token, "^[0-9]+$")) |>
  count(token, sort = TRUE)

tokens |>
  slice_max(n, n=20) |> 
  ggplot(aes(x = reorder(token, n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Words After Removing Stopwords", x = "Words", y = "Frequency")

# wordcloud2()
```

I now see more informative words such as prodecedure, pain, anesthesia,
words that have relevance to this topic.

------------------------------------------------------------------------

## Question 4: ngrams

Repeat question 2, but this time tokenize into bi-grams. How does the
result change if you look at tri-grams? Note we need to remove stopwords
a little differently. You don’t need to recreate the wordclouds.

``` r
sw_start <- paste0("^", paste(stop_words_custom, collapse=" |^"), "$")
sw_end <- paste0("", paste(stop_words_custom, collapse="$| "), "$")

tokens_bigram <- mt_samples |>
  select(transcription) |>
  unnest_tokens(bigram, transcription, token = "ngrams", n = 2) |>
  filter(!str_detect(bigram, sw_start)) |>
  filter(!str_detect(bigram, sw_end)) |> 
  count(bigram, sort = TRUE)

tokens_bigram |> 
  slice_max(n, n = 20) |> 
  ggplot(aes(x = reorder(bigram, n), y = n)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Bigrams", x = "Bigrams", y = "Frequency")
```

This gives us a much clearer picture as to what the most used terms are
in this data. Things like “blood loss” and “0 vicryl” are medical terms
that make a lot more sense now that we are using bigrams.

------------------------------------------------------------------------

## Question 5: Examining words

Using the results from the bigram, pick a word and count the words that
appear before and after it, and create a plot of the top 20.

``` r
library(stringr)
# e.g. patient, blood, preoperative...
tokens_bigram |>
  filter(str_detect(bigram, paste0("\\b", "blood", "\\b"))) |>
    mutate(word = str_remove(bigram, "blood"),
         word = str_remove_all(word, " ")) |> 
  slice_max(n, n=20) |> 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = paste("Top 20 Bigrams Containing 'blood'", sep = ""), x = "Bigrams", y = "Frequency")
```

------------------------------------------------------------------------

## Question 6: Words by Specialties

Which words are most used in each of the specialties? You can use
`group_by()` and `top_n()` from `dplyr` to have the calculations be done
within each specialty. Remember to remove stopwords. How about the 5
most used words?

``` r
mt_samples |>
  unnest_tokens(word, transcription) |>
  filter(!word %in% stop_words_custom) |>
  count(medical_specialty, word, sort = TRUE) |>
  group_by(medical_specialty) |> 
  slice_max(n, n=1, with_ties=FALSE)

mt_samples |>
  unnest_tokens(word, transcription) |>
  filter(!word %in% stop_words_custom) |>
  count(medical_specialty, word, sort = TRUE) |>
  group_by(medical_specialty) |>
  slice_max(n, n = 5, with_ties = FALSE)
```

## Question 7: Topic Models

See if there are any themes in the data by using a topic model (LDA).

- you first need to create a document term matrix
- then you can try the LDA function in `topicmodels`. Try different k
  values.
- create a facet plot of the results from the LDA (see code from
  lecture)

``` r
transcripts_dtm <- mt_samples |>
  select(transcription) |>
  unnest_tokens(token, transcription) |>
  filter(!token %in% stop_words_custom) |>
  DocumentTermMatrix()


transcripts_dtm <- as.matrix(transcripts_dtm)   

transcripts_lda <- LDA(transcripts_dtm, k = 5, control = list(seed = 1234))
transcripts_lda

top_terms <- tidy(transcripts_lda, matrix = "beta") |>
  group_by(topic) |>
  slice_max(beta, n = 10) |> 
  ungroup() |>
  arrange(topic, -beta)

top_terms |>
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```
