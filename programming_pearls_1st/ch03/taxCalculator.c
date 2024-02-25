#include <stdio.h>

#define element_of(a) (sizeof(a)/sizeof(a[0]))

typedef struct IncomeTaxInfo
{
	float taxThreshold;
	float baseTax;
	float rate;
} IncomeTaxInfo;

int compareByTaxThreshold(const void* l, const void* r)
{
	const IncomeTaxInfo* left = ( const IncomeTaxInfo*)l;
	const IncomeTaxInfo* right = ( const IncomeTaxInfo*)r;

	return (left->taxThreshold > right->taxThreshold) - (left->taxThreshold < right->taxThreshold);
}

static const IncomeTaxInfo taxInfoMap[] = {
	{0, 0, 0},
	{2200, 0, .14},
	{2700, 70, .15},
	{3200, 145, .16},
	{3700, 225, .17},
	{4200, 310, .18},
};

// return the last element that is no larger than key
// It has not done any sanity check, make sure inputs are valid 
const void* bsearch_self(const void* key, const void* ptr, size_t elementCount, size_t elementSize,
		int (*comp)(const void*, const void*))
{
	size_t start = 0;
	size_t end = elementCount - 1;

	while(start<=end)
	{
		size_t mid = start + (end-start)/2;
		if(comp(ptr+mid*elementSize, key) <= 0) {
			start = ++mid;
		} else {
			end = --mid;
		}
	}

	end = (end <= 0) ? 0 : end;
	return ptr+end*elementSize;
}

// It will act abnormally when you have input non-numberic characters

int main()
{
	IncomeTaxInfo input;

	while (1) {
		printf("Enter your income >>> ");
		scanf("%f", &input.taxThreshold);

		if(input.taxThreshold <= 0.0f)
			break;

		const IncomeTaxInfo* taxInfo = (const IncomeTaxInfo*)bsearch_self(&input, taxInfoMap, 
							element_of(taxInfoMap), sizeof(IncomeTaxInfo), compareByTaxThreshold);

		float tax = taxInfo->baseTax + (input.taxThreshold-taxInfo->taxThreshold)*taxInfo->rate;
		printf("Your tax: %f\n", tax);

	}

	return 0;
}

